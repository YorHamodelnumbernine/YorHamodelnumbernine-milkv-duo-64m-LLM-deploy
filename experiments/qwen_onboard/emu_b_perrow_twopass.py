#!/usr/bin/env python3
"""emu_b_perrow_twopass.py — Path B (direct per-ch INT8) + per-row INT8 act
+ engine-realistic two-pass rshift (KG=128), the A'/B common-item quality test.

emu_perrow.py uses a naive single-pass rshift=12 which we confirmed is TOO lossy
(A' 0/3, B 1/3).  The real on-board engine uses the two-pass matmul with
data-adaptive pass2 rshift (emu_wsdq.matmul_wsdq twopass).  This test applies
the same engine-realistic two-pass to Path B weights to get the TRUE verdict
for the per-row common item on B.

Path B weights: direct per-channel INT8 quantized from bf16 (no INT4 detour).
Reference next_token (bf16): 2130 / 12095 / 99366 (same as A').
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import emu_perrow as EP
import emu_wsdq as W
from emu_wsdq import forward_wsdq, PROMPTS, REF

KAL = os.path.join(HERE, "weights_kal")


def main():
    print("[B_perrow_twopass] loading Path B weights (per-ch INT8 from bf16)...", flush=True)
    # load_weights_B returns (embed, esc, frms, layers) with layers = engine format
    # (rms_attn, mats_int8, rms_ffn, lsc, bias)
    _, _, _, layers = EP.load_weights_B()

    # reference embed/final_rms from K-aligned dir (same int8 per-row embed)
    embed = np.fromfile(os.path.join(KAL, "embed_i8.bin"), dtype=np.int8).reshape(EP.V, EP.D)
    esc = np.fromfile(os.path.join(KAL, "embed_scales.f32"), dtype=np.float32)
    frms = np.fromfile(os.path.join(KAL, "final_rms.f32"), dtype=np.float32)

    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(os.path.join(HERE, "..", "qwen_int4", "model"),
                                            trust_remote_code=True)
    except Exception as e:
        print("tokenizer fail:", e)

    passed = 0
    res = {"tag": "B_perrow_twopass128", "prompts": {}}
    for p in PROMPTS:
        ids = tok.encode(p, add_special_tokens=False) if tok else [0]
        logits = forward_wsdq(ids, embed, esc, frms, layers, "twopass")
        t5 = np.argsort(-logits)[:5].tolist()
        v5 = logits[t5].tolist()
        gap = v5[0] - v5[1] if len(v5) > 1 else 0.0
        ok = t5[0] == REF[p]
        passed += ok
        res["prompts"][p] = {"n_tokens": len(ids), "next": t5[0], "ref": REF[p],
                             "ok": bool(ok), "top5": t5, "gap": float(gap)}
        print(f"[B_perrow_twopass] '{p}' next={t5[0]} ref={REF[p]} "
              f"{'OK' if ok else 'MISMATCH'} gap={gap:.3f}", flush=True)
    res["verdict"] = f"{passed}/3"
    print(f"[B_perrow_twopass] ============ VERDICT {passed}/3 ============", flush=True)
    with open(os.path.join(HERE, "b_perrow_twopass_ref.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("[B_perrow_twopass] wrote b_perrow_twopass_ref.json")


if __name__ == "__main__":
    main()
