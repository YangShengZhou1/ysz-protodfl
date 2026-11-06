"""Small python-docx helpers shared by the delivery reports (Chinese text)."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


def set_run(run, size=10.5, bold=False, east="宋体", latin="Times New Roman"):
    run.font.size = Pt(size)
    run.bold = bold
    run.font.name = latin
    run._element.rPr.rFonts.set(qn("w:eastAsia"), east)


def new_doc() -> Document:
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21.0), Cm(29.7)
    section.top_margin = section.bottom_margin = Cm(2.4)
    section.left_margin = section.right_margin = Cm(2.6)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    return doc


def title(doc, text: str, size=17):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run(p.add_run(text), size=size, bold=True, east="黑体")
    p.paragraph_format.space_after = Pt(4)
    return p


def subtitle(doc, text: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run(p.add_run(text), size=9)
    p.paragraph_format.space_after = Pt(10)
    return p


def h1(doc, text: str):
    p = doc.add_paragraph()
    set_run(p.add_run(text), size=13, bold=True, east="黑体")
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(5)
    return p


def h2(doc, text: str):
    p = doc.add_paragraph()
    set_run(p.add_run(text), size=11, bold=True, east="黑体")
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    return p


def para(doc, text: str, indent=True, size=10.5, space_after=4):
    p = doc.add_paragraph()
    set_run(p.add_run(text), size=size)
    p.paragraph_format.line_spacing = 1.35
    p.paragraph_format.space_after = Pt(space_after)
    if indent:
        p.paragraph_format.first_line_indent = Pt(size * 2)
    return p


def bullet(doc, text: str, size=10.5):
    p = doc.add_paragraph()
    set_run(p.add_run("· " + text), size=size)
    p.paragraph_format.line_spacing = 1.35
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.left_indent = Pt(12)
    return p


def table(doc, headers, rows, widths=None, size=9):
    tbl = doc.add_table(rows=1, cols=len(headers))
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for idx, text in enumerate(headers):
        cell = tbl.rows[0].cells[idx]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_run(p.add_run(text), size=size, bold=True, east="黑体")
    for row in rows:
        cells = tbl.add_row().cells
        for idx, text in enumerate(row):
            cells[idx].text = ""
            p = cells[idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            set_run(p.add_run(str(text)), size=size)
    if widths:
        for row in tbl.rows:
            for idx, width in enumerate(widths):
                row.cells[idx].width = Cm(width)
    return tbl


def caption(doc, text: str):
    p = doc.add_paragraph()
    set_run(p.add_run(text), size=9)
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(8)
    return p


def figure(doc, path: Path, cap: str, width=15.0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Cm(width))
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    caption(doc, cap).alignment = WD_ALIGN_PARAGRAPH.CENTER


def pp(value: float, sign=True) -> str:
    if value != value:
        return "-"
    return f"{value:+.2f}" if sign else f"{value:.2f}"
