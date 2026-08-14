#!/usr/bin/env python3
"""deploy_qwen_weights.py — push weights_kal/ to Duo /data/qwen/ with md5 verify.

scp -O equivalent via paramiko pipe (dropbear).  Per-file push + md5 check.
Usage: python3 deploy_qwen_weights.py [--only layer0_kal.bin] [--dir weights_kal]
"""
import os, sys, hashlib, time, paramiko

HOST = "192.168.42.1"
USER = "root"
PASS = "milkv"
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, sys.argv[sys.argv.index("--dir") + 1] if "--dir" in sys.argv else "weights_kal")
DST = "/data/qwen"
ONLY = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None

def md5_local(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=10, look_for_keys=False, allow_agent=False)
    ssh.exec_command(f"mkdir -p {DST}")
    time.sleep(0.5)

    files = sorted(f for f in os.listdir(SRC) if os.path.isfile(os.path.join(SRC, f)))
    if ONLY:
        files = [f for f in files if f == ONLY]
    t_tot = time.time()
    n_ok = 0
    for i, fn in enumerate(files):
        lp = os.path.join(SRC, fn)
        data = open(lp, "rb").read()
        lmd5 = hashlib.md5(data).hexdigest()
        t0 = time.time()
        stdin, stdout, stderr = ssh.exec_command(
            f"cat > {DST}/{fn}", timeout=600)
        CHUNK = 65536
        for j in range(0, len(data), CHUNK):
            stdin.write(data[j:j + CHUNK])
        stdin.channel.shutdown_write()
        stdout.channel.recv_exit_status()
        _, o2, _ = ssh.exec_command(f"md5sum {DST}/{fn}")
        rmd5 = o2.read().decode().split()[0]
        dt = time.time() - t0
        ok = rmd5 == lmd5
        n_ok += ok
        rate = len(data) / 1e6 / max(dt, 1e-9)
        print(f"[{'OK ' if ok else 'FAIL'}] {fn} {len(data)/1e6:.2f}MB {rate:.1f}MB/s", flush=True)
        if not ok:
            print(f"  local={lmd5} remote={rmd5}", flush=True)
    print(f"deployed {n_ok}/{len(files)} in {time.time()-t_tot:.1f}s")
    ssh.close()

if __name__ == "__main__":
    main()
