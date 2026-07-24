#!/usr/bin/env python3
"""Push a large file to Duo in chunks via paramiko, then extract and run the demo.
Usage: python3 deploy_smollm2.py
"""
import paramiko, os, sys, time, hashlib, struct

HOST = "192.168.42.1"
CHUNK = 32768  # 32KB chunks for reliable transfer
WEIGHTS_TAR = "/tmp/smollm2_weights.tar"
TEST_TOKENS = "/tmp/test_tokens.bin"
BINARY = os.path.expanduser("~/Documents/MilkV_duo_project/tpu_bench/smollm2_demo")

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

def push_file(ssh, local_path, remote_path):
    """Push a single file via cat > remote."""
    data = open(local_path, "rb").read()
    local_md5 = hashlib.md5(data).hexdigest()
    size = len(data)
    print(f"  Pushing {os.path.basename(local_path)} ({size/1024/1024:.1f} MB)...", end=" ", flush=True)

    # For larger files, push in chunks via separate cat commands
    stdin, stdout, stderr = ssh.exec_command(
        f"cat > {remote_path} && md5sum {remote_path}", timeout=300)

    for i in range(0, size, CHUNK):
        stdin.write(data[i:i + CHUNK])
    stdin.channel.shutdown_write()

    md5_out = stdout.read().decode(errors="replace")
    stdout.channel.recv_exit_status()
    remote_md5 = md5_out.split()[0] if md5_out else "?"

    if remote_md5 == local_md5:
        print(f"OK ({size} bytes)")
        return True
    else:
        print(f"FAIL local={local_md5} remote={remote_md5}")
        return False

def main():
    print("=" * 60)
    print("SmolLM2-135M Deployment to Milk-V Duo")
    print("=" * 60)

    # 1. Connect
    print("\n[1/5] Connecting to Duo...")
    ssh = ssh_connect()
    print("  Connected!")

    # Check disk space
    out, err, rc = ssh_exec(ssh, "df -h /tmp | tail -1; free -m | head -2")
    print(f"  Disk: {out.strip()}")

    # 2. Create weight directory and push weights
    print("\n[2/5] Pushing weight files...")
    ssh_exec(ssh, "mkdir -p /tmp/smollm2")

    # Check if weights already exist (by checking embed.i8 size)
    out, _, _ = ssh_exec(ssh, "wc -c < /tmp/smollm2/embed.i8 2>/dev/null || echo 0")
    embed_sz = int(out.strip())
    expected_sz = 49152 * 576  # vocab * d_model = 28,311,552

    if embed_sz == expected_sz:
        print(f"  Weight files already exist (embed.i8={embed_sz} bytes). Skip upload.")
    else:
        if embed_sz > 0:
            print(f"  Partial weights found ({embed_sz} vs {expected_sz}), re-uploading...")
        # Push tarball
        if not push_file(ssh, WEIGHTS_TAR, "/tmp/smollm2_weights.tar"):
            print("ERROR: Failed to push weights tarball")
            sys.exit(1)

        # Extract
        print("  Extracting weights...")
        out, err, rc = ssh_exec(ssh, "cd /tmp && rm -rf smollm2 && tar xf smollm2_weights.tar && rm smollm2_weights.tar && ls /tmp/smollm2/config.bin", timeout=120)
        if rc != 0:
            print(f"ERROR: Extract failed: {err}")
            sys.exit(1)
        print(f"  Extract OK. Files in /tmp/smollm2/: {out.strip()}")

    # 3. Push binary
    print("\n[3/5] Pushing smollm2_demo...")
    if not push_file(ssh, BINARY, "/tmp/smollm2_demo"):
        print("ERROR: Failed to push binary")
        sys.exit(1)
    ssh_exec(ssh, "chmod +x /tmp/smollm2_demo")

    # 4. Push test tokens
    print("\n[4/5] Pushing test tokens...")
    if not push_file(ssh, TEST_TOKENS, "/tmp/test_tokens.bin"):
        print("ERROR: Failed to push test tokens")
        sys.exit(1)

    # 5. Run inference
    print("\n[5/5] Running inference...")
    out, err, rc = ssh_exec(ssh, "timeout 600 /tmp/smollm2_demo /tmp/test_tokens.bin 8", timeout=600)
    if out: print(out)
    if err: print(f"[STDERR]\n{err}")
    print(f"\n[RC={rc}]")

    ssh.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
