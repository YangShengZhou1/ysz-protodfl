"""Remote helper for the 3090 box: run a shell command or push a file.

Credentials live in ``secrets/server_access.txt`` and are read at runtime,
never stored in this file.

Usage:
    python scripts/remote_exec.py --cmd "nvidia-smi"
    python scripts/remote_exec.py --put local.py /remote/path/file.py
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
import time

import paramiko

CONNECT_RETRIES = 3
CONNECT_BACKOFF = 5


def load_credentials() -> tuple[str, int, str, str]:
    root = pathlib.Path(__file__).resolve().parents[3]
    path = root / "secrets" / "server_access.txt"
    text = path.read_text(encoding="utf-8")

    def grab(pattern: str) -> str:
        match = re.search(pattern, text)
        if not match:
            raise SystemExit(f"cannot parse {pattern!r} from {path}")
        return match.group(1)

    host = grab(r"服务器[：:]\s*(\S+)")
    port = int(grab(r"端口[：:]\s*(\d+)"))
    user = grab(r"用户[：:]\s*(\S+)")
    password = grab(r"密码[：:]\s*(\S+)")
    return host, port, user, password


def connect() -> paramiko.SSHClient:
    host, port, user, password = load_credentials()
    last_error: Exception | None = None
    for attempt in range(1, CONNECT_RETRIES + 1):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                host,
                port=port,
                username=user,
                password=password,
                timeout=20,
                banner_timeout=30,
                auth_timeout=30,
            )
            return client
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            client.close()
            if attempt < CONNECT_RETRIES:
                time.sleep(CONNECT_BACKOFF)
    raise SystemExit(f"ssh failed after {CONNECT_RETRIES} attempts: {last_error}")


def run_command(client: paramiko.SSHClient, command: str, timeout: int = 120) -> int:
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    stdin.close()
    out = stdout.read()
    err = stderr.read()
    status = stdout.channel.recv_exit_status()
    sys.stdout.buffer.write(out)
    sys.stdout.buffer.flush()
    sys.stderr.buffer.write(err)
    sys.stderr.buffer.flush()
    return status


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", help="shell command to run remotely")
    parser.add_argument("--put", nargs=2, metavar=("LOCAL", "REMOTE"), help="upload a file")
    parser.add_argument("--get", nargs=2, metavar=("REMOTE", "LOCAL"), help="download a file")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    if not args.cmd and not args.put and not args.get:
        parser.error("one of --cmd, --put or --get is required")

    client = connect()
    try:
        if args.put:
            local, remote = args.put
            sftp = client.open_sftp()
            try:
                sftp.put(local, remote)
                sys.stdout.write(f"uploaded {local} -> {remote}\n")
            finally:
                sftp.close()
        if args.get:
            remote, local = args.get
            pathlib.Path(local).parent.mkdir(parents=True, exist_ok=True)
            sftp = client.open_sftp()
            try:
                sftp.get(remote, local)
                sys.stdout.write(f"downloaded {remote} -> {local}\n")
            finally:
                sftp.close()
        if args.cmd:
            return run_command(client, args.cmd, timeout=args.timeout)
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
