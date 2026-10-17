#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chat_qwen.py — Milk-V Duo Qwen2 (INT8/INT4) CHAT 模式 host 侧问答包装.

工作流 (与 TPU 侧 CHAT 模式对齐):
  用户文本 --tokenize(add_special_tokens=False)--> input_tokens.bin
    --duo_push 推板--> /data/qwen/input_tokens.bin
    --板上引擎 (CHAT=1 VERIFY=0 RSH=1, 出货配置 mmap+GSC_ION 默认) 跑推理-->
    引擎输出 "CHAT: <token_id> <token_id> ..." 到远端 log
    --duo_pull 拉 log--> 解析 CHAT: 行 --> de-tokenize --> 打印回答

input_tokens.bin 格式 (与 TPU 约定):
  [n_tokens int32 小端][token ids int32 小端...]

CHAT: 输出格式 (与 TPU 约定):
  一行 "CHAT:" 后接空格分隔的生成 token id (可多行, 全部拼接)。

用法:
  ../qwen_int4/.venv/bin/python3 chat_qwen.py                    # 交互循环
  ../qwen_int4/.venv/bin/python3 chat_qwen.py "中国的首都是"       # 一次性问答
  python3 chat_qwen.py --max-new 8 "问题"                        # 指定生成上限
  python3 chat_qwen.py --dry-run "问题"                          # 只生成本地输入, 不推板不跑

说明:
  - 脚本会自动用 ../qwen_int4/.venv/bin/python3 重启自身 (tokenizer 依赖该 venv)。
  - 引擎 CHAT 输出 = prefill next(首个生成 token) + decode 步输出, 故回答 token 数 = max_new+1。
  - KV 上限默认取引擎当前 KV_CAP=48 (TPU 已由 20 提升); 若 TPU 再调整, 用 --kv-cap 对齐。
"""

import argparse
import os
import re
import struct
import subprocess
import sys
import time

# --------------------------------------------------------------------------
# 路径常量
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/Documents/MilkV_duo_project")
PRE = os.path.join(HERE, "..", "qwen_int4")
VENV_PY = os.path.join(PRE, ".venv", "bin", "python3")
DEFAULT_MODEL = os.path.join(PRE, "model")

DUO_SSH = os.path.join(ROOT, "duo_ssh.py")
DUO_PUSH = os.path.join(ROOT, "duo_push.py")
DUO_PULL = os.path.join(ROOT, "duo_pull.py")

# 板端常量 (与 TPU 实现对齐, 以 TPU 为准)
REMOTE_DIR_DEFAULT = "/data/qwen"
REMOTE_TOKENS_NAME = "input_tokens.bin"
REMOTE_BIN_DEFAULT = "qwen_engine_lmhead2_aprime"   # TPU 的 CHAT 版二进制名
KV_CAP_DEFAULT = 48        # 引擎 KV cache 容量 (TPU 已由 20 提升到 48, +0.67MB DDR anon)
PREFILL_CAP_DEFAULT = 7    # 单 chunk prefill 上限 (引擎 chunked prefill 每 chunk<=7)
MAX_NEW_DEFAULT = 16

# 出货配置: LW_READ=mmap + GSC_ION 启用 + VERIFY=0 + RSH=1
# 引擎 CHAT 模式生成 min(DECODE_STEPS, KV_CAP - N) 个 token, 故 host 用 DECODE_STEPS 控制长度
#
# 注意 (2026-08-16 更新): GSC_ION=1 已在引擎侧修复并上板验证 — 新增 carveout 预检,
#   空间不足干净回退冷 mmap, 不再进入 SDK mem_alloc_raw 重试 -> SIGABRT / ION 泄漏;
#   pool/mp 分配全 NULL 检查 + 部分释放。实测 CHAT+GSC_ION=1 稳定 (exit=0, ION 退出归零)。
#   默认用 1 (出货配置); 需要更保守时 --gsc-ion 0 (mmap 冷读, 不占 ION)。
# 另: CHAT=1 已跳过 3-prompt 回归, 直接 input_tokens.bin -> prefill -> decode。
ENGINE_ENV_FMT = "CHAT=1 VERIFY=0 RSH=1 LW_READ=mmap GSC_ION={gsc_ion} DECODE_STEPS={max_new}"

# 板上跑前按惯例开 swap (standing instruction)
SWAP_CMD = (
    "if ! swapon --show 2>/dev/null | grep -q '/swap'; then "
    "if [ ! -f /swap ]; then dd if=/dev/zero of=/swap bs=1M count=64 2>/dev/null; fi; "
    "mkswap /swap >/dev/null 2>&1; swapon /swap 2>/dev/null || true; fi; "
    "echo '--- swap ---'; swapon --show 2>/dev/null; free -m | head -2"
)

# 估算: decode 每 token ~11.3s (mmap 出货锚点) + prefill/加载开销
SECS_PER_TOKEN_EST = 15
PREFILL_FIXED_EST = 90


def _venv_relaunch():
    """若当前解释器不是 qwen_int4 的 venv, 用 venv python 重启自身.

    chat_qwen.py 依赖 transformers(AutoTokenizer), 只在 ../qwen_int4/.venv 存在。
    这样无论用户用哪个 python3 调用, 都能保证 tokenizer 环境正确。
    """
    if os.path.realpath(sys.executable) == os.path.realpath(VENV_PY):
        return
    if os.path.exists(VENV_PY):
        os.execv(VENV_PY, [VENV_PY] + sys.argv)
    # venv 不存在: 不阻断, 让后续 import transformers 自然报错并给出提示


_venv_relaunch()


# --------------------------------------------------------------------------
# 远端控制 (复用 duo_push / duo_ssh / duo_pull)
# --------------------------------------------------------------------------
def _run(args, timeout=60, verbose=True):
    """调用一个 duo_* 脚本, 返回 (rc, stdout, stderr)."""
    if verbose:
        print(f"[host] {' '.join(args)}", file=sys.stderr)
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout + 30)
    return p.returncode, p.stdout, p.stderr


def duo_ssh(cmd, timeout=60, verbose=True):
    """在 Duo 上执行命令; 返回远端退出码 (从 [RC=n] 解析)."""
    rc, out, err = _run([sys.executable, DUO_SSH, cmd, str(timeout)], timeout, verbose)
    m = re.findall(r"\[RC=(-?\d+)\]", out)
    remote_rc = int(m[-1]) if m else None
    if verbose:
        if out:
            for line in out.splitlines():
                if line.strip():
                    print(f"[remote] {line}", file=sys.stderr)
        if err:
            print(err, file=sys.stderr, end="")
    return remote_rc if remote_rc is not None else rc


def duo_push(local, remote, verbose=True):
    """推文件到 Duo; 失败抛异常."""
    rc, out, err = _run([sys.executable, DUO_PUSH, local, remote], 120, verbose)
    if rc != 0:
        raise RuntimeError(f"duo_push 失败 rc={rc}: {err} {out}")
    return out.strip()


def duo_pull(remote, local, verbose=True):
    """从 Duo 拉文件; 失败抛异常."""
    rc, out, err = _run([sys.executable, DUO_PULL, remote, local], 120, verbose)
    if rc != 0:
        raise RuntimeError(f"duo_pull 失败 rc={rc}: {err} {out}")
    return out.strip()


# --------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------
_tokenizer = None


def get_tokenizer(model_path):
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        print(f"[tok] 加载 tokenizer: {model_path}", file=sys.stderr)
        _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"[tok] vocab={_tokenizer.vocab_size} eos={_tokenizer.eos_token_id} "
              f"pad={_tokenizer.pad_token_id}", file=sys.stderr)
    return _tokenizer


# --------------------------------------------------------------------------
# 输入编码 / input_tokens.bin
# --------------------------------------------------------------------------
def encode_text(tok, text):
    """按 add_special_tokens=False 编码, 返回 int32 id 列表."""
    ids = tok.encode(text, add_special_tokens=False)
    return list(ids)


def write_tokens_bin(ids, path):
    """写 [n_tokens int32][ids int32...] (小端), 与 qwen_tokenize.py / TPU 一致."""
    buf = struct.pack("<i", len(ids)) + struct.pack("<%di" % len(ids), *ids)
    with open(path, "wb") as f:
        f.write(buf)
    return buf


def check_limits(n_prompt, max_new, kv_cap, prefill_cap):
    """返回 (block_msg_or_None, [warning_msgs]).

    引擎 CHAT 生成 min(DECODE_STEPS, KV_CAP - N) 个 token:
    - 硬限: prompt 本身 >= KV_CAP -> 引擎直接报错 (需留 >=1 个 decode slot).
    - 软限: max_new > KV_CAP - N -> 引擎会把 decode 步数截断到 KV_CAP - N.
    - 说明: prompt > prefill_cap -> 引擎走 chunked prefill (每 chunk<=7), host 放行.
    """
    block = None
    warns = []
    room = max(0, kv_cap - n_prompt)
    if n_prompt >= kv_cap:
        block = (
            f"提示词过长：输入 {n_prompt} 个 token，已等于/超过当前 KV 容量上限 "
            f"{kv_cap}（引擎要求至少留 1 个 decode slot）。\n"
            f"  → 请缩短问题，或确认 TPU 是否已上调 KV_CAP 后用 --kv-cap 对齐。"
        )
    elif max_new > room:
        warns.append(
            f"注意：按 KV 容量 {kv_cap}，引擎实际最多运行 {room} 个 decode 步 "
            f"(prompt {n_prompt} 已占位)；--max-new={max_new} 会被截断。"
        )
    if n_prompt > prefill_cap:
        warns.append(
            f"提示：提示词 {n_prompt} 个 token 将走 chunked prefill "
            f"(每 chunk ≤{prefill_cap})，由引擎内部自动分块。"
        )
    return block, warns


# --------------------------------------------------------------------------
# CHAT: 输出解析
# --------------------------------------------------------------------------
_CHAT_RE = re.compile(r"^\s*CHAT:\s*(.*)$")


def parse_chat_log(log_text):
    """从引擎 log 解析所有 'CHAT:' 行, 返回生成的 token id 列表."""
    ids = []
    for line in log_text.splitlines():
        m = _CHAT_RE.match(line)
        if not m:
            continue
        for tok in m.group(1).split():
            tok = tok.strip()
            if not tok:
                continue
            try:
                ids.append(int(tok))
            except ValueError:
                # 容忍 CHAT: 行尾跟了非数字诊断(例如总结行), 跳过该碎片
                print(f"[warn] CHAT 行含非数字 {tok!r}，跳过", file=sys.stderr)
    return ids


def strip_special_ids(tok, ids):
    """去掉 eos/pad 等特殊 token id (151643 <|endoftext|> / 151645 <|im_end|>)."""
    specials = {tok.eos_token_id, tok.pad_token_id, tok.bos_token_id}
    specials.discard(None)
    return [i for i in ids if i not in specials]


# --------------------------------------------------------------------------
# 单轮问答
# --------------------------------------------------------------------------
def run_qa(args, text, tok):
    t0 = time.time()
    ids = encode_text(tok, text)
    if not ids:
        print("(空输入)", file=sys.stderr)
        return

    n = len(ids)
    print(f"[tok] {text!r} -> {n} tokens: {ids[:16]}{'...' if n > 16 else ''}",
          file=sys.stderr)

    # ---- 限制检查 (中文提示) ----
    block, warns = check_limits(n, args.max_new, args.kv_cap, args.prefill_cap)
    for w in warns:
        print(f"[warn] {w}", file=sys.stderr)
    if block:
        print("[限制] " + block.replace("\n", "\n       "), file=sys.stderr)
        return

    # ---- 本地写 input_tokens.bin ----
    local_bin = os.path.join(HERE, REMOTE_TOKENS_NAME)
    buf = write_tokens_bin(ids, local_bin)
    print(f"[tok] 已写 {local_bin} ({len(buf)} B, {n} ids)", file=sys.stderr)

    if args.dry_run:
        env_s = ENGINE_ENV_FMT.format(max_new=args.max_new, gsc_ion=args.gsc_ion)
        print(f"[dry-run] 推板目标: {args.remote_dir}/{REMOTE_TOKENS_NAME}", file=sys.stderr)
        print(f"[dry-run] 引擎命令: {env_s} ./{args.bin}  (远端 {args.remote_dir})",
              file=sys.stderr)
        print(f"[dry-run] 期望输出: CHAT: " + " ".join(f"<id{i}>" for i in range(args.max_new + 1)),
              file=sys.stderr)
        return

    remote_tokens = f"{args.remote_dir}/{REMOTE_TOKENS_NAME}"

    # ---- swap (按惯例) ----
    if not args.no_swap:
        duo_ssh(SWAP_CMD, timeout=60)

    # ---- 推板 ----
    duo_push(local_bin, remote_tokens)

    # ---- 跑引擎 (重定向到远端 log, 避免 SSH 长连接占用 stdout) ----
    ts = time.strftime("%Y%m%d_%H%M%S")
    remote_log = f"{args.remote_dir}/chat_{ts}.log"
    bin_ref = args.bin if args.bin.startswith("/") else f"./{args.bin}"
    env_s = ENGINE_ENV_FMT.format(max_new=args.max_new, gsc_ion=args.gsc_ion)
    run_cmd = (
        f"cd {args.remote_dir} && {env_s} {bin_ref} "
        f"> {remote_log} 2>&1; echo CHAT_EXIT=$?"
    )
    est = PREFILL_FIXED_EST + args.max_new * SECS_PER_TOKEN_EST
    timeout = args.timeout if args.timeout else int(est)
    print(f"[engine] 运行 {args.bin} (预计 ~{est}s, SSH timeout={timeout}s)...",
          file=sys.stderr)
    run_rc = duo_ssh(run_cmd, timeout=timeout)

    # ---- 拉 log ----
    local_log = os.path.join(HERE, f"chat_{ts}.log")
    duo_pull(remote_log, local_log)
    log_text = open(local_log, "r", errors="replace").read()

    # ---- 解析 CHAT: ----
    gen_ids = parse_chat_log(log_text)
    print(f"[parse] CHAT: 行解析出 {len(gen_ids)} 个 token id, "
          f"engine_exit={run_rc}", file=sys.stderr)
    if run_rc != 0:
        print(f"[error] 引擎退出码 {run_rc}，log 尾部:", file=sys.stderr)
        for line in log_text.splitlines()[-20:]:
            print(f"  {line}", file=sys.stderr)
    if not gen_ids:
        print(f"[error] log 中未找到 CHAT: 行 ({remote_log})", file=sys.stderr)
        for line in log_text.splitlines()[-20:]:
            print(f"  {line}", file=sys.stderr)
        if not args.keep_log:
            os.unlink(local_log)
        return

    # ---- de-tokenize ----
    display_ids = strip_special_ids(tok, gen_ids) if args.skip_special else gen_ids
    answer = tok.decode(display_ids, skip_special_tokens=args.skip_special)
    elapsed = time.time() - t0
    print(f"\n[回答 {elapsed:.0f}s] {answer}\n")
    if args.verbose:
        print(f"[detail] gen_ids={gen_ids}", file=sys.stderr)

    if not args.keep_log:
        os.unlink(local_log)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Milk-V Duo Qwen2 CHAT host 侧问答包装 (交互/一次性)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("text", nargs="?", default=None,
                    help="一次性问答的问题文本; 省略则进入交互循环")
    ap.add_argument("--bin", default=REMOTE_BIN_DEFAULT,
                    help=f"板上引擎二进制名 (默认 {REMOTE_BIN_DEFAULT})")
    ap.add_argument("--remote-dir", default=REMOTE_DIR_DEFAULT,
                    help=f"板上工作目录 (默认 {REMOTE_DIR_DEFAULT})")
    ap.add_argument("--max-new", type=int, default=MAX_NEW_DEFAULT,
                    help=f"decode 步数 = 生成 token 数-1 (默认 {MAX_NEW_DEFAULT}; "
                         f"回答 token 数 = max_new+1, 含 prefill next)")
    ap.add_argument("--kv-cap", type=int, default=KV_CAP_DEFAULT,
                    help=f"KV cache 容量上限 (默认 {KV_CAP_DEFAULT}, 以 TPU 实现为准)")
    ap.add_argument("--prefill-cap", type=int, default=PREFILL_CAP_DEFAULT,
                    help=f"单 chunk prefill 上限 (默认 {PREFILL_CAP_DEFAULT})")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="tokenizer 模型目录")
    ap.add_argument("--timeout", type=int, default=None,
                    help="SSH 命令超时(秒); 默认按 max-new 估算")
    ap.add_argument("--no-swap", action="store_true",
                    help="跳过板上 swap 前置")
    ap.add_argument("--keep-log", action="store_true",
                    help="保留拉回的本地 log")
    ap.add_argument("--no-skip-special", action="store_true",
                    help="de-tokenize 时保留特殊 token (默认跳过)")
    ap.add_argument("--dry-run", action="store_true",
                    help="只生成本地 input_tokens.bin 并打印命令, 不推板不跑")
    ap.add_argument("--gsc-ion", choices=["0", "1"], default="1",
                    help="GSC_ION 开关 (默认 1=ION 缓存层权重, 出货配置; 引擎已修 OOM/泄漏, 见脚本头注释; 0=mmap 冷读不占 ION)")
    ap.add_argument("--verbose", action="store_true",
                    help="输出调试细节")
    args = ap.parse_args()

    args.skip_special = not args.no_skip_special

    # 一次性模式: 显式给文本
    if args.text is not None:
        if not args.text.strip():
            print("(空问题)", file=sys.stderr)
            return 1
        tok = get_tokenizer(args.model)
        run_qa(args, args.text, tok)
        return 0

    # 交互模式
    tok = get_tokenizer(args.model)
    print("=" * 56, file=sys.stderr)
    print("  Milk-V Duo Qwen2 — CHAT (host 侧包装)", file=sys.stderr)
    print(f"  引擎: {args.remote_dir}/{args.bin}  max_new={args.max_new}", file=sys.stderr)
    env_s = ENGINE_ENV_FMT.format(max_new=args.max_new, gsc_ion=args.gsc_ion)
    print(f"  KV_CAP={args.kv_cap} prefill_cap={args.prefill_cap} "
          f"({env_s})", file=sys.stderr)
    print("  输入 'quit'/'exit'/'q' 或 Ctrl+C 退出", file=sys.stderr)
    print("=" * 56, file=sys.stderr)

    while True:
        try:
            text = input("你: ")
        except (EOFError, KeyboardInterrupt):
            print("\n[bye]", file=sys.stderr)
            break
        if text.strip().lower() in ("quit", "exit", "q"):
            break
        if not text.strip():
            continue
        run_qa(args, text, tok)

    return 0


if __name__ == "__main__":
    sys.exit(main())
