"""Extract text from legacy PDFs and TeX sources, then list KL-related passages."""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT.parents[1] / "archive/01_论文与答辩"
OUT = ROOT / "outputs/experiments/2026-11-05_supplement/_legacy_text"

PDFS = {
    "icme2025": ARCHIVE / "Prototype-Based_Communication_Topology_Optimization_for_Decentralized_Federated_Learning.pdf",
    "thesis": ARCHIVE / "毕业论文-终稿.pdf",
}
TEXS = sorted((ARCHIVE / "论文").rglob("*.tex"))


def dump_pdf(path: Path, target: Path) -> str:
    doc = pymupdf.open(str(path))
    text = "\n".join(page.get_text() for page in doc)
    target.write_text(text, encoding="utf-8")
    return text


def passages(text: str, pattern: str, width: int = 260, limit: int = 40) -> list[str]:
    out = []
    for match in re.finditer(pattern, text, re.I):
        seg = text[max(0, match.start() - width): match.start() + width]
        out.append(" ".join(seg.split()))
        if len(out) >= limit:
            break
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, path in PDFS.items():
        if not path.exists():
            print(f"MISSING {path}")
            continue
        text = dump_pdf(path, OUT / f"{name}.txt")
        hits = passages(text, r"KL|Kullback", limit=8)
        print("=" * 40, name, len(text), "chars,", len(hits), "passages")
        for seg in hits:
            print("  -", seg[:420])

    for tex in TEXS:
        text = tex.read_text(encoding="utf-8", errors="ignore")
        plain = re.sub(r"\s+", " ", text)
        hits = passages(plain, r"KL divergence|D_\{\\mathrm\{KL\}\}|Kullback", limit=6)
        if not hits:
            continue
        print("=" * 30, tex.name, len(hits), "passages")
        for seg in hits:
            print("  -", seg[:420])


if __name__ == "__main__":
    main()
