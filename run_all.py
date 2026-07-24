#!/usr/bin/env python3
"""Push all TPU benchmark binaries to Duo and run them, collecting results."""
import os, sys, paramiko, time

DUO_HOST = "192.168.42.1"
DUO_USER = "root"
DUO_PASS = "milkv"
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))

# List of (local_path, display_name, expected_result_keyword)
tests = []
for root, dirs, files in os.walk(BENCH_DIR):
    for f in sorted(files):
        path = os.path.join(root, f)
        if os.access(path, os.X_OK) and not f.endswith('.c') and not f.endswith('.h'):
            rel = os.path.relpath(path, BENCH_DIR)
            tests.append((path, rel))

print(f"Found {len(tests)} test binaries")

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(DUO_HOST, username=DUO_USER, password=DUO_PASS, timeout=10,
          look_for_keys=False, allow_agent=False)

# Push all binaries
import hashlib, base64
for local_path, rel_name in tests:
    data = open(local_path, "rb").read()
    b64 = base64.b64encode(data).decode()
    chunks = [b64[i:i+4096] for i in range(0, len(b64), 4096)]

    stdin, stdout, stderr = c.exec_command(
        f"cat > /tmp/{os.path.basename(rel_name)}.b64 && base64 -d /tmp/{os.path.basename(rel_name)}.b64 > /tmp/{os.path.basename(rel_name)} && chmod +x /tmp/{os.path.basename(rel_name)}",
        timeout=120)
    for chunk in chunks:
        stdin.write(chunk + '\n')
    stdin.channel.shutdown_write()
    rc = stdout.channel.recv_exit_status()
    if rc == 0:
        print(f"  PUSH OK: {os.path.basename(rel_name)}")
    else:
        print(f"  PUSH FAIL (rc={rc}): {os.path.basename(rel_name)}")

print("\n=== Running tests ===\n")
results = []
for local_path, rel_name in tests:
    name = os.path.basename(rel_name)
    stdin, stdout, stderr = c.exec_command(f"/tmp/{name} 2>&1", timeout=30)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    rc = stdout.channel.recv_exit_status()

    # Parse result
    full = out + err
    passed = "OK" if (" ALL OK" in full or "OK  " in full or "WORKS!" in full or "[OK]" in full) and "FAIL" not in full and "MISMATCH" not in full else "FAIL"

    # extract timing
    time_str = "N/A"
    for line in full.split('\n'):
        if 'time:' in line:
            time_str = line.split('time:')[1].strip().replace(' us','')
            break

    print(f"  [{passed}] {rel_name} ({time_str}us)")
    results.append((rel_name, passed, time_str))

print(f"\n=== Summary: {sum(1 for _,p,_ in results if p=='OK')}/{len(results)} pass ===")
for rel, p, t in results:
    print(f"  [{p}] {rel} ({t}us)")

c.close()
