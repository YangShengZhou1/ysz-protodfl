"""Compare the working copy against the original student code package.

Purpose: produce an auditable list of what changed since the handover
package, so every experiment result can be traced back to a known code
state. Prints a per-directory summary, then modified / added / removed
file lists with line deltas; ``--detail`` prints the unified diff for one
file.

Text files only (.py/.sh/.md/.txt/.json/.yaml/.cfg/.proto/.log are
hashed, everything else is ignored).
"""

import argparse
import difflib
import hashlib
import os
from pathlib import Path

TEXT_SUFFIXES = {
    ".py", ".sh", ".md", ".txt", ".json", ".yaml", ".yml", ".cfg",
    ".proto", ".bat", ".ps1", ".csv",
}
SKIP_DIR_NAMES = {
    "__pycache__", ".git", ".venv", "outputs", "outputs_backup_2026-11-04",
    "outputs_backup_2026-11-04-long", "node_modules", ".idea", ".ipynb_checkpoints",
}
# Directories holding datasets / archives: never part of the code audit.
SKIP_TOP_DIRS = {"data", "datacifar10"}


def collect(root):
    """os.walk with pruning so deep output trees never hit MAX_PATH."""
    root = Path(root)
    files = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIR_NAMES and not d.startswith("outputs_backup")
        ]
        if Path(dirpath) == root:
            dirnames[:] = [d for d in dirnames if d not in SKIP_TOP_DIRS]
        for name in filenames:
            path = Path(dirpath) / name
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            files[path.relative_to(root).as_posix()] = hashlib.sha256(raw).hexdigest()
    return files


def read_lines(path):
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def line_delta(original_file, current_file):
    old = read_lines(original_file)
    new = read_lines(current_file)
    added = removed = 0
    for line in difflib.unified_diff(old, new, lineterm="", n=0):
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return added, removed


def summarize(original_root, current_root):
    old = collect(original_root)
    new = collect(current_root)
    old_keys, new_keys = set(old), set(new)

    modified = sorted(k for k in old_keys & new_keys if old[k] != new[k])
    added = sorted(new_keys - old_keys)
    removed = sorted(old_keys - new_keys)
    return modified, added, removed


def print_summary(original_root, current_root, modified, added, removed):
    def bucket(paths):
        table = {}
        for p in paths:
            top = p.split("/")[0] if "/" in p else "(root)"
            table[top] = table.get(top, 0) + 1
        return table

    print(f"modified: {len(modified)}  added: {len(added)}  removed: {len(removed)}")
    for label, paths in (("MODIFIED", modified), ("ADDED", added), ("REMOVED", removed)):
        table = bucket(paths)
        if table:
            print(f"\n[{label} per directory]")
            for key in sorted(table):
                print(f"  {table[key]:4d}  {key}")

    if modified:
        print("\n[MODIFIED files, added/removed lines]")
        for rel in modified:
            add, rem = line_delta(Path(original_root) / rel, Path(current_root) / rel)
            print(f"  +{add:<4d} -{rem:<4d}  {rel}")


def print_detail(original_root, current_root, rel, context=2):
    old_lines = read_lines(Path(original_root) / rel)
    new_lines = read_lines(Path(current_root) / rel)
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"original/{rel}", tofile=f"current/{rel}",
        lineterm="", n=context,
    )
    for line in diff:
        print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--detail", default=None, help="relative path to diff verbosely")
    parser.add_argument("--list", choices=["modified", "added", "removed"], default=None)
    parser.add_argument("--context", type=int, default=2)
    args = parser.parse_args()

    if args.detail:
        print_detail(args.original, args.current, args.detail, args.context)
        return

    modified, added, removed = summarize(args.original, args.current)
    if args.list == "modified":
        for rel in modified:
            print(rel)
        return
    if args.list == "added":
        for rel in added:
            print(rel)
        return
    if args.list == "removed":
        for rel in removed:
            print(rel)
        return
    print_summary(args.original, args.current, modified, added, removed)


if __name__ == "__main__":
    main()
