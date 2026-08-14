#!/usr/bin/env python3
"""phase7_analyze_logs.py — 解析 Phase7/A' 上板 log, 输出对比表 + 验收线 + 回归清单.

[Phase 8 回归工具链] 本脚本与 phase7f_aprime_run.sh 一起构成 A'/B-2 回归工具链:
  板上跑 phase7f_aprime_run.sh (B 基线 mmap / A perf ion_db / A corr VERIFY=1),
  host 侧用本脚本输出对比表 + 验收线(9.5s)自动判定 + 回归清单.
  Phase 8 已收口: 出货配置 = LW_READ=mmap GSC_ION=1 (11.29s/token 锚点), ion_db 不达
  验收线(负优化)仅作回归对照. 本脚本锚点/验收线常量应随出货口径维护.

用法:
  python3 phase7_analyze_logs.py [dir=.]
  python3 phase7_analyze_logs.py dir=logs/

dir 内可选 log (自动识别, 不全则跳过对应行):
  layer_read_bench.log    读路径归因表 (legacy Phase 7)
  decode_mmap.log / decode_mmap_ra.log / decode_pread.log   legacy Phase 7 三档
  decode_b_mmap.log        A' B 基线: LW_READ=mmap GSC_ION=1 VERIFY=0 (锚点 11.29s)
  decode_a_iondb.log       A' perf:   LW_READ=ion_db VERIFY=0 (验收线 9.5s)
  decode_a_iondb_v1.log    A' corr:   LW_READ=ion_db VERIFY=1 (回归清单)
  ion_after.log            板上 A' 跑完后 ION debugfs (used/rate) + stale 进程数

验收口径 (A' 立项):
  1. decode A' 中位数 ~8s, 全部档 <10s, 验收线 9.5s (相对 11.29s 基线 -16~29%)
  2. 回归红线: NEXT 3/3, bad1=bad2=r_opt=rsh=0, min_gap>=0.05, VmSwap 稳态,
     prefetch_n~=23xN (L-1/步), err=0, sd_wait 小, ION 无泄漏.
"""
import re, os, sys

ANCHOR_AVG = 11.29          # decode_e1_v0.log 锚点 (Phase 7e GSC_ION=1 生产口径)
ACCEPT_LINE = 9.5           # A' 验收线 (CEO 立项)
L = 24                      # 层数; A' 每步 issue L-1 = 23 次预读
PF_PER_STEP = L - 1         # = 23 (层 L-1 无后继层)

# ---- 解析单个 decode log ----
def parse_decode(path):
    if not os.path.exists(path):
        return None
    txt = open(path, encoding="utf-8", errors="replace").read()
    r = {"path": path, "next": ("?", "?"), "tius": "?", "bit": ("?", "?", "?", "?"),
         "avg": None, "steps": [], "gaps": [], "vms": [], "aprime": None,
         "ion_alloc_fail": False, "mode": "?"}
    m = re.search(r"24L regression: expected_next (\d+)/(\d+)", txt)
    if m: r["next"] = m.groups()
    m = re.search(r"24L regression: TIU internal (\S+)", txt)
    if m: r["tius"] = m.group(1)
    # decode 段 bit-exact (4 项, 兼容旧版无 rsh) — 优先 decode 行, 回退 P1/P2 行
    b = None
    m = re.search(r"decode bit-exact: bad1=(\d+) bad2=(\d+) r_opt=(\d+) rsh=(\d+)", txt)
    if m: b = m.groups()
    if not b:
        m = re.search(r"decode bit-exact: bad1=(\d+) bad2=(\d+) r_opt=(\d+)", txt)
        if m: b = m.groups() + ("?",)
    if not b:
        m = re.search(r"P1/P2 bit-exact: bad1=(\d+) bad2=(\d+)\s+r_opt mismatches=(\d+)\s+rsh\(scan-vs-table\)=(\d+)", txt)
        if m: b = m.groups()
    if not b:
        m = re.search(r"P1/P2 bit-exact: bad1=(\d+) bad2=(\d+)\s+r_opt mismatches=(\d+)", txt)
        if m: b = m.groups() + ("?",)
    if b: r["bit"] = b
    m = re.search(r"decode avg per-token = ([\d.]+)s over (\d+) steps", txt)
    if m: r["avg"] = (float(m.group(1)), int(m.group(2)))
    for line in txt.splitlines():
        m = re.search(r"DECODE pos=\d+ .*?gap=([\d.]+).*?total=([\d.]+)s VmSwap=(\d+)", line)
        if m:
            r["steps"].append(float(m.group(2)))
            r["gaps"].append(float(m.group(1)))
            r["vms"].append(int(m.group(3)))
    # A' metrics (每步打印, 取最后一行即累计终值)
    ap = [re.search(r"A' metrics: prefetch_n=(\d+) err=(\d+) sd=([\d.]+)s memcpy=([\d.]+)s sd_wait=([\d.]+)s",
                    ln) for ln in txt.splitlines()]
    ap = [m for m in ap if m]
    if ap:
        m = ap[-1]
        r["aprime"] = {"pf_n": int(m.group(1)), "err": int(m.group(2)),
                       "sd": float(m.group(3)), "memcpy": float(m.group(4)),
                       "sd_wait": float(m.group(5))}
    r["ion_alloc_fail"] = ("ION alloc" in txt and "FAILED" in txt) or ("ion ioctl fail" in txt)
    m = re.search(r"layer read mode = ([^|]+?)(?:\s*\|)", txt)
    if m: r["mode"] = m.group(1).strip()
    m = re.search(r"\[LW_ION_DB\] SD_BUF_A/B=(\d+) B x2 in ION carveout, bounce=(\d+) B, warmup\(0\) done", txt)
    r["sd_buf"] = (int(m.group(1)), int(m.group(2))) if m else None
    return r

def fmt_bit(b):
    return "%s/%s/%s/%s" % b

def check_next(r):
    return r and r["next"] == ("3", "3")

def check_bit(r):
    """bit-exact: bad1=bad2=r_opt=0 必需; rsh 若有则也须为 0 (legacy log 无 rsh 则忽略)."""
    if not r:
        return False
    b = r["bit"]
    if len(b) != 4 or any(x == "?" for x in b[:3]):
        return False
    if b[3] == "?":
        return b[0] == "0" and b[1] == "0" and b[2] == "0"
    return all(x == "0" for x in b)

def check_min_gap(r):
    return r and bool(r["gaps"]) and min(r["gaps"]) >= 0.05

def check_vmswap(r):
    """VmSwap 稳态: 峰值 <= 4000kB 且 max-min <= 1000kB (无 swap 抖动)."""
    if not r or not r["vms"]:
        return False
    lo, hi = min(r["vms"]), max(r["vms"])
    return hi <= 4000 and (hi - lo) <= 1000

def check_prefetch(r):
    """prefetch_n ~= 23×steps (L-1/步), err=0. 取累计终值."""
    if not r or not r["aprime"] or not r["avg"]:
        return None
    _, n = r["avg"]
    exp = PF_PER_STEP * n
    a = r["aprime"]
    return {"exp": exp, "pf_n": a["pf_n"], "err": a["err"],
            "ok": a["err"] == 0 and abs(a["pf_n"] - exp) <= 3}

def check_sd_wait(r):
    """sd_wait 语义: 主线程等待预读完成的累计时间.

    SD-bound 时 (SD 读 ~9.5s > 计算 ~4.1s), sd_wait≈SD读-重叠计算 属正常;
    compute-bound 时 (预读追上) sd_wait≈0. 因此不设"必须小"红线, 只做 sanity:
    sd_wait 不可能超过线程总读时间 (g_t_sd_read), 也不可能超过层总耗时.
    """
    if not r or not r["aprime"] or not r["steps"]:
        return None
    a = r["aprime"]
    tot = sum(r["steps"])
    sane = a["sd_wait"] <= a["sd"] + 0.01 and a["sd_wait"] <= tot + 0.01
    # 隐含 SD-bound floor: 每步线程读时间 + 层0 冷 sync(0.4s) + LM head(0.5s)
    n = len(r["steps"])
    floor = a["sd"] / n + 0.9 if n else float("nan")
    return {"sd_wait": a["sd_wait"], "sd_read": a["sd"], "tot_layers": tot,
            "ok": sane, "floor_est": floor}

def check_ion_leak(ion_path):
    """ION 无泄漏: used==0 (或 usage rate 0%), 且无 stale 进程."""
    if not os.path.exists(ion_path):
        return None
    txt = open(ion_path, encoding="utf-8", errors="replace").read()
    used = re.search(r"used:(\d+) bytes", txt)
    rate = re.search(r"usage rate:(\d+)%", txt)
    used_v = int(used.group(1)) if used else -1
    rate_v = int(rate.group(1)) if rate else -1
    stale = re.search(r"^(\d+)\s*$", txt, re.M)
    stale_n = int(stale.group(1)) if stale else -1
    ok = (used_v >= 0 and used_v < 200000) or (rate_v >= 0 and rate_v < 2)
    if stale_n >= 0:
        ok = ok and stale_n == 0
    return {"used": used_v, "rate": rate_v, "stale": stale_n, "ok": ok}

def main():
    d = "."
    if len(sys.argv) > 1 and sys.argv[1].startswith("dir="):
        d = sys.argv[1].split("=", 1)[1]
    J = lambda n: os.path.join(d, n)

    # ---- 归因表 (legacy, 可选) ----
    if os.path.exists(J("layer_read_bench.log")):
        print("=" * 78)
        print("读路径吞吐归因表 (layer_read_bench, 201MB/pass, cold)")
        print("=" * 78)
        for line in open(J("layer_read_bench.log"), encoding="utf-8", errors="replace"):
            if re.match(r"^(mmap_|pread_|read_|mode |\(SD )", line.strip()):
                print(line.rstrip())
        print()

    # ---- A' A/B 对比 ----
    print("=" * 78)
    print("A' A/B 对比: ion_db vs mmap 基线 (锚点 %.2fs, 验收线 %.1fs)" % (ANCHOR_AVG, ACCEPT_LINE))
    print("=" * 78)
    hdr = "%-20s %8s %5s %9s %8s %16s %9s %s" % (
        "mode", "avg(s)", "N", "min_gap", "NEXT", "bad1/bad2/r_opt/rsh",
        "VmSwap", "A' pf/err/sd_wait")
    print(hdr); print("-" * 78)

    # 锚点基线
    print("%-20s %8.2f %5s %9s %8s %16s %9s %s" % (
        "mmap 锚点(7e)", ANCHOR_AVG, "-", "0.50", "3/3", "0/0/0/0", "2852", "-"))
    # 本轮 A/B logs
    modes = [
        ("decode_b_mmap.log",    "mmap B(本轮)"),
        ("decode_a_iondb.log",   "ion_db A(perf)"),
        ("decode_a_iondb_v1.log","ion_db A(corr)"),
    ]
    present = []
    for fn, label in modes:
        r = parse_decode(J(fn))
        if not r:
            print("%-20s (log 缺失)" % label); continue
        present.append((label, fn, r))
        a, n = r["avg"] if r["avg"] else (float("nan"), 0)
        gap = min(r["gaps"]) if r["gaps"] else float("nan")
        vms = max(r["vms"]) if r["vms"] else -1
        ap = r["aprime"]
        ap_s = ("%d/%d/%.2f" % (ap["pf_n"], ap["err"], ap["sd_wait"])) if ap else "-"
        a_s = "?" if a != a else "%8.2f" % a   # nan -> ?
        print("%-20s %8s %5d %9s %8s %16s %7s %s" % (
            label, a_s, n,
            "%.4f" % gap if gap == gap else "?", "/".join(r["next"]),
            fmt_bit(r["bit"]), "%dkB" % vms if vms >= 0 else "?", ap_s))
        for i, t in enumerate(r["steps"]):
            print("    step %2d: %8.2fs  gap=%.4f  VmSwap=%dkB%s" % (
                i, t, r["gaps"][i], r["vms"][i],
                "  [A' metrics]" if r["aprime"] and i == len(r["steps"]) - 1 else ""))

    # legacy 三档 (可选, 兼容旧 log)
    legacy = []
    for mode in ["mmap", "mmap_ra", "pread"]:
        r = parse_decode(J("decode_%s.log" % mode))
        if r:
            legacy.append((mode, r))
    for mode, r in legacy:
        a, n = r["avg"] if r["avg"] else (float("nan"), 0)
        gap = min(r["gaps"]) if r["gaps"] else float("nan")
        vms = max(r["vms"]) if r["vms"] else -1
        a_s = "?" if a != a else "%8.2f" % a
        print("%-20s %8s %5d %9s %8s %16s %7s %s" % (
            mode, a_s, n,
            "%.4f" % gap if gap == gap else "?", "/".join(r["next"]),
            fmt_bit(r["bit"]), "%dkB" % vms if vms >= 0 else "?", "-"))

    # ---- 验收线 (A perf) ----
    a_log = None
    for label, fn, r in present:
        if "perf" in label:
            a_log = (label, r)
    print()
    print("=" * 78)
    print("验收线自动判定 (A' perf, 验收线 %.1fs)" % ACCEPT_LINE)
    print("=" * 78)
    if a_log:
        label, r = a_log
        if r["avg"]:
            a, n = r["avg"]
            ok_avg = a < ACCEPT_LINE
            med = sorted(r["steps"])[len(r["steps"]) // 2] if r["steps"] else float("nan")
            ok_all_lt10 = bool(r["steps"]) and all(t < 10.0 for t in r["steps"])
            delta = (a - ANCHOR_AVG) / ANCHOR_AVG * 100.0
            print("  A' avg       : %8.2fs/token  %s" % (a, "PASS (<9.5)" if ok_avg else "FAIL (>=9.5)"))
            print("  全档 <10s    : %s  (max %.2fs)" % ("PASS" if ok_all_lt10 else "FAIL",
                                                       max(r["steps"]) if r["steps"] else float("nan")))
            print("  中位数       : ~%.2fs (预期 ~8s)" % med)
            print("  vs 锚点 11.29: %+.1f%%" % delta)
            print("  => 验收线 %s" % ("PASS" if ok_avg and ok_all_lt10 else "FAIL"))
        else:
            print("  A' perf log 缺 decode avg 行 — 无法判定")
    else:
        print("  未找到 decode_a_iondb.log (A perf)")

    # ---- 回归清单 (A corr VERIFY=1) ----
    print()
    print("=" * 78)
    print("回归清单 (A corr, LW_READ=ion_db VERIFY=1)")
    print("=" * 78)
    c = None
    for label, fn, r in present:
        if "corr" in label:
            c = (label, r)
    if c:
        label, r = c
        def row(name, ok, detail):
            print("  [%s] %-22s %s" % ("PASS" if ok else "FAIL", name, detail))
        row("NEXT 3/3", check_next(r), "/".join(r["next"]))
        row("bit-exact bad1=bad2=r_opt=rsh=0", check_bit(r), fmt_bit(r["bit"]))
        row("decode min_gap >= 0.05", check_min_gap(r),
            ("min=%.4f" % min(r["gaps"])) if r["gaps"] else "no steps")
        row("VmSwap 稳态", check_vmswap(r),
            ("peak=%dkB delta=%dkB" % (max(r["vms"]), max(r["vms"]) - min(r["vms"]))) if r["vms"] else "no vms")
        pf = check_prefetch(r)
        if pf:
            row("prefetch_n~=23xN err=0", pf["ok"],
                "pf_n=%d exp≈%d err=%d" % (pf["pf_n"], pf["exp"], pf["err"]))
        else:
            row("prefetch_n~=23xN err=0", False, "A' metrics 缺失")
        sw = check_sd_wait(r)
        if sw:
            row("sd_wait sanity", sw["ok"],
                "sd_wait=%.3fs sd_read=%.3fs layers=%.2fs | 隐含SD-bound floor≈%.2fs/step"
                % (sw["sd_wait"], sw["sd_read"], sw["tot_layers"], sw["floor_est"]))
        else:
            row("sd_wait sanity", False, "A' metrics 缺失")
        row("ION 无泄漏 (log 内)", not r["ion_alloc_fail"],
            "无 alloc fail" if not r["ion_alloc_fail"] else "检测到 ION alloc/ioctl fail")
        if r["sd_buf"]:
            print("  [info] SD_BUF_A/B = %d B x2 in ION carveout, bounce=%d B" % r["sd_buf"])
    else:
        print("  未找到 decode_a_iondb_v1.log (A corr) — 回归清单跳过")

    # ---- ION 无泄漏 (ion_after.log) ----
    il = check_ion_leak(J("ion_after.log"))
    if il:
        print()
        print("=" * 78)
        print("ION 无泄漏检查 (ion_after.log, A' 全部 run 退出后)")
        print("=" * 78)
        print("  [%s] used=%d bytes rate=%d%% stale_proc=%d" % (
            "PASS" if il["ok"] else "FAIL", il["used"], il["rate"], il["stale"]))
    else:
        print()
        print("ION 无泄漏检查: ion_after.log 缺失 (由 phase7_deploy_run_host.sh aprime 模式自动拉取)")

    # ---- 总体 ----
    print()
    print("=" * 78)
    if not a_log and not c:
        overall = None
        verdict = "N/A — 缺 A' log (先跑 sh phase7_deploy_run_host.sh aprime 再分析)"
    else:
        overall = True
        if a_log and a_log[1]["avg"]:
            a = a_log[1]["avg"][0]
            overall = overall and a < ACCEPT_LINE
        if c:
            r = c[1]
            overall = overall and check_next(r) and check_bit(r) and check_min_gap(r) \
                and check_vmswap(r) and (check_prefetch(r) and check_prefetch(r)["ok"]) \
                and (check_sd_wait(r) and check_sd_wait(r)["ok"]) and not r["ion_alloc_fail"]
        if il:
            overall = overall and il["ok"]
        verdict = "PASS — A' 验收 + 回归全过" if overall else "FAIL — 存在未过项, 见上"
    print("总体判定: %s" % verdict)
    print("=" * 78)

if __name__ == "__main__":
    main()
