"""Run a training command with idle/hard timeouts and process-tree cleanup.

The watchdog treats both redirected console output and log-directory updates as
activity. If nothing changes for ``--idle-timeout`` seconds, or the hard limit is
reached, the entire child process tree is terminated and a JSON status file is
written.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def _activity_marker(paths):
    latest = 0.0
    for path in paths:
        if not path.exists():
            continue
        try:
            latest = max(latest, path.stat().st_mtime)
        except OSError:
            continue
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_file():
                    try:
                        latest = max(latest, child.stat().st_mtime)
                    except OSError:
                        continue
    return latest


def _kill_tree(proc):
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idle-timeout", type=float, default=600)
    parser.add_argument("--hard-timeout", type=float, default=7200)
    parser.add_argument("--poll", type=float, default=2)
    parser.add_argument("--watch-dir", action="append", default=[])
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--stdout", required=True)
    parser.add_argument("--stderr", required=True)
    parser.add_argument("--cwd", default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command after --")

    stdout_path = Path(args.stdout).resolve()
    stderr_path = Path(args.stderr).resolve()
    status_path = Path(args.status_file).resolve()
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    watch_dirs = [Path(p).resolve() for p in args.watch_dir]

    start = time.time()
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    with open(stdout_path, "wb", buffering=0) as stdout_file, open(stderr_path, "wb", buffering=0) as stderr_file:
        proc = subprocess.Popen(
            command,
            cwd=args.cwd,
            stdout=stdout_file,
            stderr=stderr_file,
            creationflags=creationflags,
            start_new_session=os.name != "nt",
        )

        reason = "completed"
        try:
            last_activity = time.time()
            while proc.poll() is None:
                now = time.time()
                if now - start > args.hard_timeout:
                    reason = "hard_timeout"
                    _kill_tree(proc)
                    break
                marker = max(
                    _activity_marker([stdout_path, stderr_path, *watch_dirs]),
                    last_activity,
                )
                if marker > last_activity:
                    last_activity = now
                if now - last_activity > args.idle_timeout:
                    reason = "idle_timeout"
                    _kill_tree(proc)
                    break
                time.sleep(args.poll)
            proc.wait()
        except KeyboardInterrupt:
            reason = "interrupted"
            _kill_tree(proc)
            proc.wait()
        finally:
            elapsed = round(time.time() - start, 2)
            status = {
                "command": command,
                "pid": proc.pid,
                "returncode": proc.returncode,
                "reason": reason,
                "elapsed_seconds": elapsed,
                "idle_timeout": args.idle_timeout,
                "hard_timeout": args.hard_timeout,
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            }
            status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(status, ensure_ascii=False, indent=2))
            if reason != "completed":
                return 124
            return int(proc.returncode or 0)


if __name__ == "__main__":
    sys.exit(main())
