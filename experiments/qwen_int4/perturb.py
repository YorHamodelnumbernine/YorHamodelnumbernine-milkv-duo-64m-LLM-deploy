#!/usr/bin/env python3
"""Gate 1: single-layer logit / matmul perturbation, INT4 vs bf16 (host).

For selected layers, with the REAL hidden state flowing through the model:
  (a) per-matrix matmul output  h_norm @ W^T : bf16 vs INT4-dequant weights
      -> max / RMS relative error  (the "will it collapse?" gate)
  (b) full-model forward with ONLY that layer quantized -> hidden state after
      the layer vs bf16 baseline, AND final-logits next_token / top5 gap drift.

Usage:
  python3 perturb.py --model model --prompt "..." [--layers 0 8 16 23] [--G 32 64 128]
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q4 import group_quantize

EPS = 1e-6
MODMAP = {
    "q_proj": ("self_attn",), "k_proj": ("self_attn",), "v_proj": ("self_attn",),
    "o_proj": ("self_attn",), "gate_proj": ("mlp",), "up_proj": ("mlp",),
    "down_proj": ("mlp",),
}


def rms_norm(x, w):
    v = x.to(torch.float32)
    rms = torch.sqrt(v.pow(2).mean(-1, keepdim=True) + EPS)
    return (v / rms) * w.to(torch.float32)


def stats(bf16_out, q_out):
    a = bf16_out.float()
    b = q_out.float()
    rms_a = a.pow(2).mean().sqrt().item()
    max_a = a.abs().max().item()
    diff = a - b
    rms_err = diff.pow(2).mean().sqrt().item()
    max_err = diff.abs().max().item()
    return dict(rms_err=rms_err, max_err=max_err,
                rel_rms=rms_err / rms_a if rms_a > 0 else float("nan"),
                rel_max=max_err / max_a if max_a > 0 else float("nan"))


def get_layer_modules(layer):
    sa, mlp = layer.self_attn, layer.mlp
    return {n: sa if n in ("q_proj", "k_proj", "v_proj", "o_proj") else mlp
            for n in MODMAP}


def get_attn_norm(layer):
    return getattr(layer, "self_attn_layer_norm",
                   getattr(layer, "input_layernorm"))


def get_ffn_norm(layer):
    return getattr(layer, "mlp_layer_norm",
                   getattr(layer, "post_attention_layernorm"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model")
    ap.add_argument("--prompt", default="今天天气很好，我们去公园")
    ap.add_argument("--layers", nargs="+", type=int, default=[0, 8, 16, 23])
    ap.add_argument("--G", nargs="+", type=int, default=[32, 64, 128])
    args = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.eval()
    print(f"[perturb] model loaded {time.time()-t0:.1f}s", flush=True)

    ids = tok(args.prompt, return_tensors="pt")
    print(f"[perturb] prompt tokens={ids['input_ids'].shape[1]} "
          f"ids={ids['input_ids'].tolist()[0][:8]}...", flush=True)

    layers = model.model.layers
    n_layer = len(layers)
    args.layers = [k for k in args.layers if 0 <= k < n_layer]

    # ---- baseline forward: capture per-layer input AND output hidden states ----
    h_in, h_out = {}, {}
    hooks = []
    for k in args.layers:
        def mk_in(kk):
            return lambda mod, a, b: h_in.__setitem__(kk, a[0].detach())
        def mk_out(kk):
            return lambda mod, a, b: h_out.__setitem__(kk, b[0].detach())
        hooks.append(layers[k].register_forward_hook(mk_in(k)))
        hooks.append(layers[k].register_forward_hook(mk_out(k)))
    with torch.no_grad():
        base = model(**ids)
    for h in hooks:
        h.remove()
    base_logits = base.logits[0, -1].float()
    base_next = base_logits.argmax().item()
    tk = torch.topk(base_logits, 5)
    print(f"[perturb] baseline next_token={base_next} top5={tk.indices.tolist()} "
          f"gap1_2={(tk.values[0]-tk.values[1]).item():.3f}", flush=True)

    report = {"model": os.path.abspath(args.model), "prompt": args.prompt,
              "prompt_tokens": ids["input_ids"].shape[1],
              "baseline_next": base_next, "baseline_top5": tk.indices.tolist(),
              "layers": args.layers, "G": args.G,
              "matmul": {}, "layer_out": {}, "logits": {}}

    for k in args.layers:
        layer = layers[k]
        attn_norm_w = get_attn_norm(layer)
        ffn_norm_w = get_ffn_norm(layer)
        h = h_in[k][0].float()
        h_attn_norm = rms_norm(h, attn_norm_w.weight)
        h_ffn_norm = rms_norm(h, ffn_norm_w.weight)
        bf16_W = {n: getattr(m, n).weight.detach().float() for n, m in get_layer_modules(layer).items()}
        ffn_act = torch.nn.functional.silu(h_ffn_norm @ bf16_W["gate_proj"].T) * (
            h_ffn_norm @ bf16_W["up_proj"].T)
        bf16_mm = {
            "q_proj": h_attn_norm @ bf16_W["q_proj"].T,
            "k_proj": h_attn_norm @ bf16_W["k_proj"].T,
            "v_proj": h_attn_norm @ bf16_W["v_proj"].T,
            "o_proj": h_attn_norm @ bf16_W["o_proj"].T,
            "gate_proj": h_ffn_norm @ bf16_W["gate_proj"].T,
            "up_proj": h_ffn_norm @ bf16_W["up_proj"].T,
            "down_proj": ffn_act @ bf16_W["down_proj"].T,
        }
        base_layer_out = h_out[k][0].float()
        report["layer_out"][str(k)] = {}

        for G in args.G:
            # quantize the 7 matrices of this layer
            dq_t = {}
            for n, W in bf16_W.items():
                Wn = W.numpy().astype(np.float32)
                _, _, _, _, dq = group_quantize(Wn, G, asym=False)
                dq_t[n] = torch.from_numpy(dq).to(torch.bfloat16)

            # (a) raw matmul perturbation
            row = {"G": G, "matrices": {}}
            for n in bf16_W:
                src = (h_attn_norm if n in ("q_proj", "k_proj", "v_proj", "o_proj")
                       else ffn_act if n == "down_proj" else h_ffn_norm)
                row["matrices"][n] = stats(bf16_mm[n], src @ dq_t[n].float().T)

            # (b) full-model forward with only this layer quantized
            saved = {}
            for n, m in get_layer_modules(layer).items():
                lin = getattr(m, n)
                saved[n] = lin.weight.data
                lin.weight.data = dq_t[n]
            cap = {}
            hook = layers[k].register_forward_hook(
                lambda mod, a, b, c=cap: c.__setitem__("out", b[0].detach()))
            with torch.no_grad():
                qout = model(**ids)
            hook.remove()
            for n, m in get_layer_modules(layer).items():
                getattr(m, n).weight.data = saved[n]

            lo = cap["out"][0].float()
            st = stats(base_layer_out, lo)
            row["layer_out_rel_rms"] = st["rel_rms"]
            row["layer_out_rel_max"] = st["rel_max"]
            report["layer_out"][str(k)][f"G{G}"] = {
                "rel_rms": st["rel_rms"], "rel_max": st["rel_max"]}

            # logits drift
            q_logits = qout.logits[0, -1].float()
            q_next = q_logits.argmax().item()
            q_tk = torch.topk(q_logits, 5)
            report["logits"][f"L{k}_G{G}"] = {
                "next": q_next, "argmax_same": q_next == base_next,
                "top5": q_tk.indices.tolist(),
                "gap12": (q_tk.values[0] - q_tk.values[1]).item(),
                "gap_base12": (tk.values[0] - tk.values[1]).item(),
                "logit_mse": ((q_logits - base_logits).pow(2).mean().item()),
                "logit_max_abs_err": (q_logits - base_logits).abs().max().item(),
            }
            report["matmul"][f"L{k}_G{G}"] = row
            print(f"[perturb] L{k} G{G}: matmul rel_rms max="
                  f"{max(v['rel_rms'] for v in row['matrices'].values()):.5f} | "
                  f"layer_out rel_rms={st['rel_rms']:.5f} | "
                  f"next={q_next} argmax_same={q_next == base_next} "
                  f"gap={q_tk.values[0]-q_tk.values[1]:.3f} (base {tk.values[0]-tk.values[1]:.3f})",
                  flush=True)

    os.makedirs("out", exist_ok=True)
    with open("out/perturb.json", "w") as f:
        json.dump(report, f, indent=1, default=float)
    print(f"[perturb] wrote out/perturb.json ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
