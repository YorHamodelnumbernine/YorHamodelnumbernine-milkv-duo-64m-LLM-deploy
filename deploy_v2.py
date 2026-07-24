#!/usr/bin/env python3
"""Deploy SmolLM2 weights via HTTP, push binary, run inference."""
import paramiko, os, sys, time, hashlib, struct

HOST = "192.168.42.1"
HTTP_URL = "http://192.168.42.237:8888/smollm2_weights.tar.gz"
WEIGHTS_DIR = "/data/smollm2"
CHUNK = 32768
BINARY = os.path.expanduser("~/Documents/MilkV_duo_project/tpu_bench/smollm2_demo")
TEST_TOKENS = "/tmp/test_tokens.bin"

def ssh_connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for attempt in range(5):
        try:
            c.connect(HOST, username="root", password="milkv", timeout=15,
                      look_for_keys=False, allow_agent=False, banner_timeout=20)
            return c
        except Exception as e:
            print(f"  SSH attempt {attempt+1}/5: {e}")
            time.sleep(5)
    raise RuntimeError("Cannot connect to Duo")

def ssh_exec(ssh, cmd, timeout=120):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    rc = stdout.channel.recv_exit_status()
    return out, err, rc

def push_small_file(ssh, local_path, remote_path):
    """Push small files (<1MB) via cat pipe."""
    data = open(local_path, "rb").read()
    local_md5 = hashlib.md5(data).hexdigest()
    size = len(data)
    print(f"  Pushing {os.path.basename(local_path)} ({size} bytes)...", end=" ", flush=True)
    stdin, stdout, stderr = ssh.exec_command(
        f"cat > {remote_path} && md5sum {remote_path}", timeout=60)
    stdin.write(data)
    stdin.channel.shutdown_write()
    md5_out = stdout.read().decode(errors="replace")
    stdout.channel.recv_exit_status()
    remote_md5 = md5_out.split()[0] if md5_out else "?"
    ok = remote_md5 == local_md5
    print("OK" if ok else f"FAIL local={local_md5} remote={remote_md5}")
    return ok

def main():
    print("=" * 60)
    print("SmolLM2-135M Deployment v2 (HTTP download)")
    print("=" * 60)

    ssh = ssh_connect()
    print("[1/6] Connected to Duo")

    # Check if weights already properly extracted
    out, _, _ = ssh_exec(ssh, f"wc -c < {WEIGHTS_DIR}/embed.i8 2>/dev/null || echo 0")
    embed_sz = int(out.strip())
    expected_sz = 49152 * 576

    if embed_sz == expected_sz:
        print(f"[2/6] Weights already exist ({WEIGHTS_DIR}/embed.i8 = {embed_sz} bytes). Skip download.")
    else:
        print(f"[2/6] Downloading weights via HTTP ({89} MB)...")
        ssh_exec(ssh, f"mkdir -p {WEIGHTS_DIR}")

        out, err, rc = ssh_exec(ssh,
            f"cd /data && wget -q --show-progress --timeout=300 -O smollm2.tar.gz '{HTTP_URL}' 2>&1",
            timeout=600)
        print(f"  wget: {out.strip()}")
        if rc != 0:
            print(f"  wget failed: {err}")
            sys.exit(1)

        print("  Extracting...")
        out, err, rc = ssh_exec(ssh,
            f"cd {WEIGHTS_DIR} && tar xzf /data/smollm2.tar.gz --strip-components=1 && rm /data/smollm2.tar.gz",
            timeout=120)
        if rc != 0:
            print(f"  Extract failed: {err}")
            sys.exit(1)
        print(f"  Extract OK")

    # Push binary (small, ~26KB)
    print("[3/6] Pushing smollm2_demo...")
    if not push_small_file(ssh, BINARY, "/tmp/smollm2_demo"):
        print("ERROR")
        sys.exit(1)
    ssh_exec(ssh, "chmod +x /tmp/smollm2_demo")

    # Push test tokens (small, 24 bytes)
    print("[4/6] Pushing test tokens...")
    if not push_small_file(ssh, TEST_TOKENS, "/tmp/test_tokens.bin"):
        print("ERROR")
        sys.exit(1)

    # Verify weights
    print("[5/6] Verifying weights...")
    out, _, _ = ssh_exec(ssh, f"ls {WEIGHTS_DIR}/config.bin {WEIGHTS_DIR}/embed.i8 {WEIGHTS_DIR}/layer0.bin")
    print(f"  {out.strip()}")

    # Run inference
    print("[6/6] Running inference...")
    out, err, rc = ssh_exec(ssh,
        f"timeout 600 /tmp/smollm2_demo /tmp/test_tokens.bin 8 --weights {WEIGHTS_DIR} 2>&1",
        timeout=600)
    if out: print(out)
    if err: print(f"[STDERR]\n{err}")
    print(f"[RC={rc}]")

    ssh.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
