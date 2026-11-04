"""Build a self-contained archive of the materials a reviewer needs.

The archive mirrors the repository tree one-to-one, so every path written in
the feedback document resolves after unzipping.  A MANIFEST.txt with SHA-256
digests is added at the root so the recipient can confirm nothing changed.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path

LONG_PREFIX = "\\\\?\\"


def long_path(path: Path) -> str:
    """Return a path usable on Windows even past the 260 character limit."""
    text = os.path.abspath(str(path))
    if os.name == "nt" and len(text) > 240 and not text.startswith(LONG_PREFIX):
        return LONG_PREFIX + text
    return text


def git(root: Path, *args: str) -> str:
    exe = os.environ.get("GIT_EXE") or "git"
    cmd = [exe, "-c", "core.longpaths=true", "-c", "core.quotepath=false", "-C", str(root), *args]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(long_path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="repository root")
    parser.add_argument("--rev", default="HEAD", help="git revision to package")
    parser.add_argument("--output", required=True, help="target zip file")
    args = parser.parse_args()

    root = Path(args.repo).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    tracked = [line for line in git(root, "ls-tree", "-r", "--name-only", args.rev).splitlines() if line]
    tracked.sort()

    missing = [rel for rel in tracked if not os.path.exists(long_path(Path(root, rel)))]
    if missing:
        print(f"missing {len(missing)} file(s) on disk:", file=sys.stderr)
        for rel in missing[:10]:
            print("  " + rel, file=sys.stderr)
        return 1

    rows: list[tuple[str, int, str]] = []
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for rel in tracked:
            path = Path(root, rel)
            size = os.stat(long_path(path)).st_size
            rows.append((rel, size, digest(path)))
            archive.write(long_path(path), arcname=rel.replace(os.sep, "/"))

        total = sum(row[1] for row in rows)
        lines = [
            "ProToDFL delivery package manifest",
            f"files: {len(rows)}",
            f"bytes: {total}",
            "",
            "sha256                                                           size  path",
        ]
        lines += [f"{h}  {s:>9}  {rel}" for rel, s, h in rows]
        archive.writestr("MANIFEST.txt", "\n".join(lines) + "\n")

    packed = output.stat().st_size
    print(f"files : {len(rows)}")
    print(f"raw   : {total / 1024 / 1024:.1f} MB")
    print(f"zip   : {packed / 1024 / 1024:.1f} MB")
    print(f"output: {output}")

    print("\ntop 5 by size:")
    for rel, size, _ in sorted(rows, key=lambda row: row[1], reverse=True)[:5]:
        print(f"  {size / 1024 / 1024:6.2f} MB  {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
