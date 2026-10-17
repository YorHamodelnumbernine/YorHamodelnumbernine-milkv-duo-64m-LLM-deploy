#!/usr/bin/env python3
"""Gate 2: full-model quality, INT4 vs bf16 for Qwen2.5-0.5B (host).

Schemes:
  rtn_sym : group INT4 RTN, symmetric (scale=max/7, clamp[-8,7]), per-group fp16 scale
  rtn_asym: group INT4 RTN, asymmetric zero-point
  awq     : AWQ-style per-input-channel scaling (from calibration act stats) + rtn_sym group quant

Metrics on eval prompts:
  next_token / argmax_same, top-5 ids, top gap, logit MSE vs bf16, KL(softmax)
Optional PPL over calibration set.

Usage:
  python3 full_model.py --calib text1.txt text2.txt --eval prompt1.txt prompt2.txt \
      [--G 32 64 128] [--schemes rtn_sym rtn_asym awq] [--embed-int4]
"""
import argparse, json, os, sys, time, math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q4 import group_quantize

EPS = 1e-6
LAYER_MODS = {k: k for k in (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")}


def get_layers(model):
    return model.model.layers


def all_linear_specs(model):
    """Yield (layer_idx, name, module) for every linear to quantize."""
    for li, layer in enumerate(get_layers(model)):
        sa, mlp = layer.self_attn, layer.mlp
        for name in LAYER_MODS:
            m = sa if name in ("q_proj", "k_proj", "v_proj", "o_proj") else mlp
            yield li, name, getattr(m, name)


def collect_calib_act(model, tokenizer, texts, max_len=64):
    """Run calibration texts, capture per-Linear input activation scales."""
    act_sum = {}
    act_cnt = {}
    handles = []
    for li, name, mod in all_linear_specs(model):
        def mk_hook(li, name):
            def hook(m, a, b):
                x = a[0].float()  # [*, in]
                key = (li, name)
                axes = tuple(range(x.dim() - 1))  # mean over all but last dim
                s = act_sum.get(key, torch.zeros(x.shape[-1]))
                act_sum[key] = s + x.abs().mean(dim=axes)
                act_cnt[key] = act_cnt.get(key, 0) + 1
            return hook
        handles.append(mod.register_forward_hook(mk_hook(li, name)))
    with torch.no_grad():
        for t in texts:
            ids = tokenizer(t, return_tensors="pt")
            if ids["input_ids"].shape[1] > max_len:
                ids["input_ids"] = ids["input_ids"][:, :max_len]
            model(**ids)
    for h in handles:
        h.remove()
    act_scales = {k: act_sum[k] / act_cnt[k] for k in act_sum}
    return act_scales


def quant_embed(E, scheme, G):
    """E: [V,D]. scheme: bf16 / int8 / int4."""
    if scheme == "bf16":
        return torch.from_numpy(E).to(torch.bfloat16)
    if scheme == "int8":
        # per-row scale (fp16)
        mx = np.max(np.abs(E), axis=1, keepdims=True)
        s = np.maximum(mx / 127.0, 1e-12)
        q = np.clip(np.round(E / s), -128, 127).astype(np.int8)
        dq = q.astype(np.float32) * s
        return torch.from_numpy(dq).to(torch.bfloat16)
    if scheme == "int4":
        # per-row (G=D) or per-group
        _, _, _, _, dq = group_quantize(E, G, asym=False)
        return torch.from_numpy(dq).to(torch.bfloat16)
    raise ValueError(scheme)


def build_quant_state(model, G, scheme, act_scales=None, alpha=0.5, embed_scheme="bf16",
                      embed_G=None):
    """Return dict (li,name)->bf16 dequantized tensor + embed weight tensor if requested."""
    state = {}
    for li, name, mod in all_linear_specs(model):
        W = mod.weight.detach().float().numpy()
        if scheme == "rtn_sym":
            _, _, _, _, dq = group_quantize(W, G, asym=False)
        elif scheme == "rtn_asym":
            _, _, _, _, dq = group_quantize(W, G, asym=True)
        elif scheme == "awq":
            s_ch = act_scales[(li, name)] ** alpha  # [in]
            Ws = W / s_ch.numpy()
            _, _, _, _, dq_s = group_quantize(Ws, G, asym=False)
            dq = dq_s * s_ch.numpy()
        else:
            raise ValueError(scheme)
        state[(li, name)] = torch.from_numpy(dq).to(torch.bfloat16)
    E = model.model.embed_tokens.weight.detach().float().numpy()  # [V,D]
    embed = quant_embed(E, embed_scheme, embed_G or G)
    return state, embed


def eval_logits(model, tokenizer, prompt):
    ids = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(**ids)
    return out.logits[0, -1].float(), ids["input_ids"].shape[1]


def report_logits(base, q, name):
    b_next = base.argmax().item()
    q_next = q.argmax().item()
    b_tk = torch.topk(base, 5)
    q_tk = torch.topk(q, 5)
    mse = (q - base).pow(2).mean().item()
    kl = F.kl_div(q.log_softmax(-1), base.softmax(-1), reduction="sum").item()
    return {
        "scheme": name,
        "base_next": b_next, "q_next": q_next, "argmax_same": q_next == b_next,
        "base_top5": b_tk.indices.tolist(), "q_top5": q_tk.indices.tolist(),
        "base_gap12": (b_tk.values[0] - b_tk.values[1]).item(),
        "q_gap12": (q_tk.values[0] - q_tk.values[1]).item(),
        "logit_mse": mse, "kl": kl,
    }


def ppl(model, tokenizer, texts, max_len=128):
    total_nll, total_tok = 0.0, 0
    with torch.no_grad():
        for t in texts:
            ids = tokenizer(t, return_tensors="pt")
            ids = ids["input_ids"][:, :max_len]
            lg = model(ids).logits[0].float()
            sh = ids[0]
            nll = F.cross_entropy(lg[:-1], sh[1:], reduction="sum").item()
            total_nll += nll
            total_tok += sh.shape[0] - 1
    return math.exp(total_nll / total_tok)


def generate(model, tokenizer, prompt, n_new, max_len=32):
    ids = tokenizer(prompt, return_tensors="pt")
    input_ids = ids["input_ids"]
    with torch.no_grad():
        for _ in range(n_new):
            lg = model(input_ids).logits[0, -1]
            nxt = lg.argmax().item()
            input_ids = torch.cat([input_ids, torch.tensor([[nxt]])], dim=1)
            if input_ids.shape[1] > max_len:
                input_ids = input_ids[:, -max_len:]
    return input_ids[0].tolist()


def apply_state(model, state, embed):
    saved = {}
    for (li, name), w in state.items():
        mod = getattr(get_layers(model)[li], "self_attn" if name in (
            "q_proj", "k_proj", "v_proj", "o_proj") else "mlp")
        lin = getattr(mod, name)
        saved[(li, name)] = lin.weight.data
        lin.weight.data = w
    saved_embed = None
    if embed is not None:
        saved_embed = model.model.embed_tokens.weight.data
        model.model.embed_tokens.weight.data = embed
    return saved, saved_embed


def restore_state(model, saved, saved_embed):
    for (li, name), w in saved.items():
        mod = getattr(get_layers(model)[li], "self_attn" if name in (
            "q_proj", "k_proj", "v_proj", "o_proj") else "mlp")
        getattr(mod, name).weight.data = w
    if saved_embed is not None:
        model.model.embed_tokens.weight.data = saved_embed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model")
    ap.add_argument("--calib", nargs="*", default=[])
    ap.add_argument("--eval", nargs="*", default=[
        "今天天气很好，我们去公园散步",
        "中国的首都是",
        "The capital of France is",
        "Translate to Chinese: hello world",
    ])
    ap.add_argument("--eval-file")
    ap.add_argument("--generate", type=int, default=0,
                    help="greedy-generate N new tokens per prompt for each scheme")
    ap.add_argument("--gen-prompts", nargs="*", default=[], help="prompts for generation")
    ap.add_argument("--G", nargs="+", type=int, default=[32, 64, 128])
    ap.add_argument("--schemes", nargs="+", default=["rtn_sym"])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--embed-scheme", default="bf16", choices=["bf16", "int8", "int4"])
    ap.add_argument("--embed-G", type=int, default=None)
    ap.add_argument("--ppl", action="store_true")
    args = ap.parse_args()
    if args.eval_file:
        with open(args.eval_file) as fh:
            args.eval = [ln.strip() for ln in fh if ln.strip()]
    args.gen_prompts = args.gen_prompts or args.eval[:2]

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.eval()
    print(f"[full_model] loaded {time.time()-t0:.1f}s", flush=True)

    # baseline logits per eval prompt
    base_logits = {}
    for i, p in enumerate(args.eval):
        lg, ntok = eval_logits(model, tok, p)
        base_logits[i] = lg
        nxt = lg.argmax().item()
        tk = torch.topk(lg, 5)
        print(f"[full_model] prompt {i}: next={nxt} top5={tk.indices.tolist()} "
              f"gap12={(tk.values[0]-tk.values[1]).item():.3f} tokens={ntok}", flush=True)

    # calibration (AWQ only)
    act_scales = None
    if "awq" in args.schemes and args.calib:
        texts = []
        for c in args.calib:
            if os.path.exists(c):
                texts.append(open(c).read())
            else:
                texts.append(c)
        print(f"[full_model] collecting calibration on {len(texts)} texts...", flush=True)
        act_scales = collect_calib_act(model, tok, texts)
        print(f"[full_model] calibration done {time.time()-t0:.1f}s", flush=True)

    results = {"base_gaps": {}, "per_prompt": {}, "generation": {}, "ppl": {},
               "config": vars(args)}

    # baseline generation reference
    if args.generate > 0:
        results["generation"]["baseline"] = {}
        for p in args.gen_prompts:
            toks = generate(model, tok, p, args.generate)
            results["generation"]["baseline"][p] = tok.decode(toks, skip_special_tokens=True)

    # baseline PPL
    ppl_texts = []
    if args.ppl:
        for c in args.calib:
            ppl_texts.append(open(c).read() if os.path.exists(c) else c)
        results["ppl"]["baseline"] = ppl(model, tok, ppl_texts)
        print(f"[full_model] baseline PPL = {results['ppl']['baseline']:.3f}", flush=True)
    for scheme in args.schemes:
        for G in args.G:
            key = f"{scheme}_G{G}"
            state, embed = build_quant_state(model, G, scheme, act_scales, args.alpha,
                                             embed_scheme=args.embed_scheme,
                                             embed_G=args.embed_G)
            saved, saved_embed = apply_state(model, state, embed)
            try:
                per = []
                for i, p in enumerate(args.eval):
                    lg, _ = eval_logits(model, tok, p)
                    per.append(report_logits(base_logits[i], lg, key))
                    print(f"[full_model] {key} prompt {i}: next={per[-1]['q_next']} "
                          f"same={per[-1]['argmax_same']} gap={per[-1]['q_gap12']:.3f} "
                          f"mse={per[-1]['logit_mse']:.2e} kl={per[-1]['kl']:.2f}", flush=True)
                results["per_prompt"][key] = per
                results["base_gaps"][key] = {
                    "argmax_same_rate": np.mean([r["argmax_same"] for r in per]),
                    "mean_gap": np.mean([r["q_gap12"] for r in per]),
                    "mean_gap_base": np.mean([r["base_gap12"] for r in per]),
                    "mean_logit_mse": np.mean([r["logit_mse"] for r in per]),
                    "mean_kl": np.mean([r["kl"] for r in per]),
                }
                if args.generate > 0:
                    results["generation"][key] = {}
                    for p in args.gen_prompts:
                        toks = generate(model, tok, p, args.generate)
                        results["generation"][key][p] = tok.decode(
                            toks, skip_special_tokens=True)
                if args.ppl and ppl_texts:
                    results["ppl"][key] = ppl(model, tok, ppl_texts)
                    print(f"[full_model] {key} PPL = {results['ppl'][key]:.3f} "
                          f"(base {results['ppl']['baseline']:.3f})", flush=True)
            finally:
                restore_state(model, saved, saved_embed)

    os.makedirs("out", exist_ok=True)
    with open("out/full_model.json", "w") as f:
        json.dump(results, f, indent=1, default=float)
    print(f"[full_model] wrote out/full_model.json ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
