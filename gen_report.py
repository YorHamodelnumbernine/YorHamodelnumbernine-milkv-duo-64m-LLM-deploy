#!/usr/bin/env python3
"""Generate HTML benchmark report for TPU tests run on Milk-V Duo (CV1800B)."""

import os, json, subprocess, sys, base64

RESULT_DIR = os.path.dirname(os.path.abspath(__file__))

TESTS = [
    # (category, op, binary, status, time_us, notes)
    ("03 Elemwise", "MUL (const x2)", "mul_const", "PASS", 559.64,
     "Input [1..16] x 2 -> [2..32]. TIU MUL INT8 with constant."),
    ("03 Elemwise", "MUL (tensor x tensor)", "mul_tensor", "PASS", 446.64,
     "Two INT8 tensors element-wise multiply."),
    ("03 Elemwise", "ADD (16-bit const 3)", "add_const", "PASS", 638.56,
     "16-bit ADD with constant 3. Split high/low byte format."),
    ("03 Elemwise", "SUB (16-bit tensor)", "sub_const", "PASS", 574.36,
     "16-bit SUB tensor - tensor. Both operands 16-bit split format."),
    ("03 Elemwise", "MAX (clamp to 8)", "max_const", "PASS", 595.88,
     "Element-wise max with constant 8. Values below 8 become 8."),
    ("03 Elemwise", "MIN (clamp to 9)", "min_const", "PASS", 648.00,
     "Element-wise min with constant 9. Values above 9 become 9."),
    ("03 Elemwise", "COPY (TIU)", "copy", "PASS", 594.72,
     "TIU tensor copy. TDMA G2L -> TIU COPY -> TDMA L2G."),
    ("03 Elemwise", "MUL QM (quantized)", "mul_qm", "FAIL", 568.32,
     "MUL with quantization multiplier. Output all zeros. Multiplier/rshift semantics unclear."),
    ("04 Logic", "AND (INT8)", "and_int8", "PASS", 589.28,
     "Bitwise AND of two INT8 tensors."),
    ("04 Logic", "OR (INT8)", "or_int8", "PASS", 594.72,
     "Bitwise OR of two INT8 tensors."),
    ("04 Logic", "XOR (INT8)", "xor_int8", "PASS", 556.68,
     "Bitwise XOR of two INT8 tensors."),
    ("04 Logic", "GE (>= const 8)", "ge_const", "PASS", 562.20,
     "Greater-than-or-equal to constant 8. Returns 1/0 (not -1/0)."),
    ("05 Shift/Copy", "Arith Shift", "arith_shift", "PASS", 602.48,
     "16-bit arithmetic right shift by -2. Sign convention: positive=left, negative=right."),
    ("06 Lookup", "Lookup Table", "lookup_table", "FAIL", 552.40,
     "cvkcv180x reports 'wrong parameter'. Likely not supported on CV180X TIU."),
    ("07 Pooling", "Max Pool (2x2, s=2)", "max_pooling", "PASS", 594.88,
     "4x4 input, 2x2 kernel, stride 2. Output [7,8,15,16]."),
    ("07 Pooling", "Avg Pool (2x2, s=2)", "avg_pooling", "PASS", 560.80,
     "4x4 input, 2x2 kernel, stride 2. avg_pooling_const=1, rshift=2."),
    ("07 Pooling", "Min Pool (2x2, s=2)", "min_pooling", "PASS", 557.52,
     "4x4 input, 2x2 kernel, stride 2. Output [1,2,9,10]."),
    ("01 Conv", "Conv 3x3 (s=1)", "conv3x3", "FAIL", 547.72,
     "cvkcv180x 'tiu conv: wrong parameter'. Conv parameter validation fails."),
    ("01 Conv", "Point-wise Conv (1x1)", "pt_conv", "FAIL", 640.52,
     "cvkcv180x 'tiu_pt_conv: wrong parameter'. PT conv not supported."),
    ("01 Conv", "Depthwise Conv (3x3)", "depthwise_conv", "FAIL", 542.08,
     "cvkcv180x 'tiu_dw_conv: wrong parameter'. DW conv not supported."),
    ("01 Conv", "Depthwise PT Conv (1x1)", "depthwise_pt_conv", "FAIL", 612.36,
     "cvkcv180x 'pt dw-conv: invalid param'. DW PT conv not supported."),
    ("02 Matmul", "Matmul (2x2)", "matmul", "FAIL", 0,
     "Assertion failed (unconditional). tiu_matrix_multiplication not implemented on CV180X."),
    ("02 Matmul", "Matmul QM (2x2)", "matmul_qm", "FAIL", 0,
     "Same assertion crash as matmul. tiu_matrix_multiplication_qm not implemented."),
]

def gen():
    n_pass = sum(1 for t in TESTS if t[3] == "PASS")
    n_total = len(TESTS)

    rows = ""
    for cat, op, _, status, time_us, notes in TESTS:
        badge = '<span class="pass">PASS</span>' if status=="PASS" else '<span class="fail">FAIL</span>'
        time_str = f'{time_us:.2f}' if time_us > 0 else 'N/A (crash)'
        rows += f"""<tr>
            <td>{cat}</td><td><strong>{op}</strong></td><td>{badge}</td>
            <td style="text-align:right">{time_str}</td>
            <td>{notes}</td>
        </tr>"""

    # Read source files
    src_files = {}
    for dirpath, dirs, files in os.walk(RESULT_DIR):
        for f in files:
            if f.endswith('.h'):
                src_files[f"common/{f}"] = open(os.path.join(dirpath, f)).read()
            elif f.endswith('.c'):
                rel = os.path.relpath(os.path.join(dirpath, f), RESULT_DIR)
                src_files[rel] = open(os.path.join(dirpath, f)).read()

    src_sections = ""
    for name in sorted(src_files.keys()):
        code = src_files[name]
        esc = code.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
        src_sections += f"""<details><summary>{name}</summary>
        <pre class="code"><code class="language-c">{esc}</code></pre></details>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Milk-V Duo CV1800B TPU Benchmark Report</title>
<style>
  body {{ font-family: -apple-system, 'Segoe UI', sans-serif; max-width:1100px; margin:2em auto; padding:0 1em; background:#0d1117; color:#c9d1d9; }}
  h1 {{ color:#58a6ff; border-bottom:2px solid #30363d; padding-bottom:0.3em; }}
  h2 {{ color:#f0883e; margin-top:1.5em; }}
  table {{ border-collapse:collapse; width:100%; margin:1em 0; }}
  th, td {{ padding:8px 12px; text-align:left; border-bottom:1px solid #30363d; }}
  th {{ background:#161b22; color:#8b949e; font-weight:600; }}
  tr:hover {{ background:#161b22; }}
  .pass {{ color:#3fb950; font-weight:bold; }}
  .fail {{ color:#f85149; font-weight:bold; }}
  .summary {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:1.2em; margin:1em 0; }}
  .code {{ background:#161b22; border-radius:6px; padding:1em; overflow-x:auto; font-size:13px; line-height:1.5; }}
  code {{ font-family:'JetBrains Mono','Fira Code',monospace; }}
  details {{ margin:0.5em 0; }}
  details summary {{ cursor:pointer; padding:4px 8px; background:#21262d; border-radius:4px; color:#58a6ff; }}
  .arch {{ color:#8b949e; font-size:0.9em; }}
  .category-pass {{ color:#3fb950; }}
  .category-fail {{ color:#f85149; }}
</style>
</head>
<body>

<h1>Milk-V Duo CV1800B TPU Benchmark Report</h1>
<p class="arch">
  Chip: CV1800B (RISC-V 64-bit) &bull; NPU Cores: 2 &times; 16 EU &bull;
  Local SRAM: 32 KB (8 banks &times; 4 KB) &bull;
  API: CVI_RT + cvikernel &bull; Date: 2026-07-19
</p>

<div class="summary">
  <strong>Summary:</strong> {n_pass}/{n_total} tests passed ({n_pass*100//n_total}%).
  Tested on real hardware (Milk-V Duo) via USB RNDIS connection.
  All timing measured via <code>CLOCK_MONOTONIC</code> around <code>CVI_RT_Submit</code>.
  Input data pre-computed at compile time; verification against pre-computed expected arrays.
</div>

<h2>Results by Category</h2>
<table>
  <tr><th>Category</th><th>Passed/Total</th><th>Status</th></tr>
  {"".join(f'<tr><td>{cat}</td><td>{sum(1 for t in TESTS if t[0]==cat and t[3]=="PASS")}/{sum(1 for t in TESTS if t[0]==cat)}</td><td class="{"category-pass" if all(t[3]=="PASS" for t in TESTS if t[0]==cat) else "category-fail"}">{"ALL PASS" if all(t[3]=="PASS" for t in TESTS if t[0]==cat) else "PARTIAL"}</td></tr>' for cat in sorted(set(t[0] for t in TESTS)))}
</table>

<h2>Results Table</h2>
<table>
  <tr><th>Category</th><th>Operation</th><th>Status</th><th>Time (us)</th><th>Notes</th></tr>
  {rows}
</table>

<h2>Key Findings</h2>

<h3>Working Operations (15/23)</h3>
<ul>
  <li><strong>INT8 Element-wise:</strong> MUL (const+tensor), MAX, MIN, COPY all work correctly. ~446-648 us.</li>
  <li><strong>INT16 Element-wise:</strong> ADD (16-bit with const), SUB (16-bit tensor) work using split high/low byte format. ~574-639 us.</li>
  <li><strong>INT8 Logic:</strong> AND, OR, XOR, GE all work correctly on INT8. ~557-595 us.</li>
  <li><strong>Arithmetic Shift:</strong> INT16 arithmetic right shift works. Shift amount sign: positive=left, negative=right. Range [-16,16]. ~602 us.</li>
  <li><strong>Pooling:</strong> Max/Avg/Min pooling (2x2 kernel, stride 2) all work. ~558-595 us.</li>
</ul>

<h3>Unsupported Operations (8/23)</h3>
<ul>
  <li><strong>Convolution family (4 ops):</strong> conv3x3, pt_conv, depthwise_conv, depthwise_pt_conv all fail with "wrong parameter" from cvikernel. The CV180X TIU appears to lack hardware support for these convolution modes, or requires specific parameter configurations not documented in the public SDK.</li>
  <li><strong>Matrix Multiplication (2 ops):</strong> Both matmul and matmul_qm crash with an unconditional assertion failure in cvikernel. The <code>tiu_matrix_multiplication</code> and <code>tiu_matrix_multiplication_qm</code> functions are stubs that immediately abort — clearly not implemented for CV180X.</li>
  <li><strong>Lookup Table:</strong> "wrong parameter" error for all table size/format combinations tested. Likely unsupported on CV180X TIU.</li>
  <li><strong>MUL QM:</strong> Executes without error but produces all-zero output. The quantization multiplier/rshift semantics are unclear.</li>
</ul>

<h3>Architecture Notes</h3>
<ul>
  <li><strong>Global memory addressing:</strong> Must use <em>absolute physical addresses</em> in <code>cvk_tg_t.start_address</code>. The <code>CVI_RT_SetBaseReg</code> + offset approach does NOT work.</li>
  <li><strong>Data coherence:</strong> CPU writes need <code>CVI_RT_MemFlush</code> before TPU execution, and <code>CVI_RT_MemInvld</code> before CPU reads.</li>
  <li><strong>INT16 format:</strong> 16-bit values use split high/low byte tensors (separate I8 tensors for high and low bytes). Little-endian byte order.</li>
  <li><strong>Timing:</strong> Single TPU submit (TDMA+TIU) takes 446-648 us for these small test cases. Overhead includes kernel launch and data transfer.</li>
  <li><strong>API:</strong> <code>CVI_RT_RegisterKernel</code> returns pointer castable to <code>cvk_context_t*</code>. Operations accessed via <code>cvk_ctx->ops->*</code>.</li>
</ul>

<h2>Source Code</h2>
{src_sections}

</body>
</html>"""

    out = os.path.join(RESULT_DIR, "report.html")
    with open(out, 'w') as f:
        f.write(html)
    print(f"Wrote {out} ({len(html)} bytes)")

if __name__ == '__main__':
    gen()
