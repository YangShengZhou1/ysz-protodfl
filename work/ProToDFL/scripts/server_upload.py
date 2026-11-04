"""Upload a local archive to the remote host via SFTP. Password from SSH_PASSWORD."""

import argparse
import os
import sys
import time

import paramiko


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--user", required=True)
    parser.add_argument("--local", required=True)
    parser.add_argument("--remote", required=True)
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
                hostname=args.host, port=args.port, username=args.user,
                password=password, timeout=30, banner_timeout=30, auth_timeout=30,
                look_for_keys=False, allow_agent=False,
            )
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    sftp = client.open_sftp()
    remote_dir = os.path.dirname(args.remote)
    if remote_dir:
        stdin, stdout, stderr = client.exec_command(f"mkdir -p {remote_dir}")
        stdout.channel.recv_exit_status()
    sftp.put(args.local, args.remote)
    sftp.close()
    client.close()
    print(f"uploaded {args.local} -> {args.user}@{args.host}:{args.remote}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
