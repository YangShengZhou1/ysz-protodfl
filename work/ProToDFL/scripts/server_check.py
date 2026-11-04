"""Read-only SSH environment check. Password must come from SSH_PASSWORD env."""

import argparse
import os
import sys
import time

import paramiko


DEFAULT_COMMANDS = [
    "hostname; whoami; uname -a",
    "nvidia-smi",
    "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader",
    "python3 --version; which python3; pip3 --version",
    "df -h .; free -h; nproc",
    "tmux -V; screen -v; conda --version; docker --version",
    "pwd; ls -la ~",
    "echo SHELL=$SHELL; echo HOME=$HOME",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--user", required=True)
    parser.add_argument("--command", action="append", default=[])
    args = parser.parse_args()

    password = os.environ.get("SSH_PASSWORD")
    if not password:
        print("SSH_PASSWORD is not set", file=sys.stderr)
        return 2

    client = None
    for attempt in range(3):
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(
                hostname=args.host,
                port=args.port,
                username=args.user,
                password=password,
                timeout=30,
                banner_timeout=30,
                auth_timeout=30,
                look_for_keys=False,
                allow_agent=False,
            )
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))

    commands = args.command or DEFAULT_COMMANDS
    for command in commands:
        print(f"\n### {command}")
        stdin, stdout, stderr = client.exec_command(command, timeout=600)
        out = stdout.read().decode("utf-8", errors="replace").strip()
        err = stderr.read().decode("utf-8", errors="replace").strip()
        if out:
            print(out)
        if err:
            print("[stderr]", err, file=sys.stderr)

    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
