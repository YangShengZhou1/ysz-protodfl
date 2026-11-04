"""Build a formatted Word report from a Markdown file.

Reusable for any Markdown note in this project: headings, tables, bullet/
numbered lists, block quotes, fenced code blocks and inline emphasis.
Defaults reproduce the original landscape experiment report; pass
``--portrait`` for reading-oriented notes.
"""

import argparse
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

DEFAULT_HEADER = "ProToDFL 有效实验记录  |  2026-11-03"
DEFAULT_TITLE = "ProToDFL 有效实验记录（2026-11-03）"


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_cell_margins(table, top=40, bottom=40, left=70, right=70):
    tbl_pr = table._tbl.tblPr
    margins = OxmlElement("w:tblCellMar")
    for side, value in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        node = OxmlElement(f"w:{side}")
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")
        margins.append(node)
    tbl_pr.append(margins)


def set_run_font(run, font_name="宋体", size=10.5, bold=None, color=None, east_asia=None):
    run.font.name = font_name
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), east_asia or font_name)
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)


def latex_to_text(text):
    text = text.replace(r"\(", "").replace(r"\)", "")
    text = text.replace(r"\[", "").replace(r"\]", "")
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
    replacements = {
        r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
        r"\mu": "μ", r"\lambda": "λ", r"\sum": "Σ", r"\times": "×",
        r"\le": "≤", r"\ge": "≥", r"\approx": "≈", r"\rightarrow": "→",
        r"\cdot": "·", r"\%": "%", r"\_": "_", r"\&": "&",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    return text


def add_inline(paragraph, text, size=10.5, base_bold=False, color=None):
    text = latex_to_text(text)
    pattern = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\*[^*]+\*)")
    for part in pattern.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2])
            set_run_font(run, size=size, bold=True, color=color)
        elif part.startswith("`") and part.endswith("`"):
            run = paragraph.add_run(part[1:-1])
            set_run_font(run, font_name="Consolas", size=size - 0.5, bold=base_bold,
                         color=color, east_asia="宋体")
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            run = paragraph.add_run(part[1:-1])
            set_run_font(run, size=size, bold=base_bold, color=color)
            run.italic = True
        else:
            run = paragraph.add_run(part)
            set_run_font(run, size=size, bold=base_bold, color=color)


def parse_table(lines):
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        rows.append(cells)
    if not rows:
        return []
    width = max(len(row) for row in rows)
    return [row + [""] * (width - len(row)) for row in rows]


def add_table(doc, rows, content_width_cm):
    if not rows:
        return
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    set_repeat_table_header(table.rows[0])
    set_cell_margins(table)

    ncols = len(rows[0])
    if ncols <= 4:
        widths = [Cm(content_width_cm / ncols)] * ncols
    else:
        first = Cm(min(3.4, content_width_cm / 4))
        rest = Cm((content_width_cm - first.cm) / (ncols - 1))
        widths = [first] + [rest] * (ncols - 1)

    for r_idx, row in enumerate(rows):
        for c_idx, value in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.width = widths[c_idx]
            cell.text = ""
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_before = Pt(0)
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.0
            add_inline(paragraph, value, size=8.3, base_bold=(r_idx == 0))
            if r_idx == 0:
                set_cell_shading(cell, "D9EAF7")
            elif r_idx % 2 == 0:
                set_cell_shading(cell, "F5F8FA")


def add_code_block(doc, code_lines, indent_cm=0.4):
    for code_line in code_lines:
        paragraph = doc.add_paragraph()
        fmt = paragraph.paragraph_format
        fmt.space_before = Pt(0)
        fmt.space_after = Pt(0)
        fmt.line_spacing = 1.0
        fmt.left_indent = Cm(indent_cm)
        run = paragraph.add_run(code_line if code_line.strip() else " ")
        set_run_font(run, font_name="Consolas", size=8.6, color=(60, 60, 60), east_asia="宋体")


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("第 ")
    set_run_font(run, size=8.5, color=(100, 100, 100))
    field_begin = OxmlElement("w:fldChar")
    field_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    field_end = OxmlElement("w:fldChar")
    field_end.set(qn("w:fldCharType"), "end")
    run._r.append(field_begin)
    run._r.append(instr)
    run._r.append(field_end)
    run2 = paragraph.add_run(" 页")
    set_run_font(run2, size=8.5, color=(100, 100, 100))


def build(source, output, header_text=DEFAULT_HEADER, title=DEFAULT_TITLE,
          subtitle="", portrait=False):
    text = Path(source).read_text(encoding="utf-8")
    lines = text.splitlines()
    doc = Document()

    section = doc.sections[0]
    if portrait:
        section.orientation = WD_ORIENT.PORTRAIT
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        section.top_margin = Cm(2.2)
        section.bottom_margin = Cm(2.0)
        section.left_margin = Cm(2.4)
        section.right_margin = Cm(2.4)
    else:
        section.orientation = WD_ORIENT.LANDSCAPE
        section.page_width = Cm(29.7)
        section.page_height = Cm(21.0)
        section.top_margin = Cm(1.6)
        section.bottom_margin = Cm(1.5)
        section.left_margin = Cm(1.7)
        section.right_margin = Cm(1.7)
    section.header_distance = Cm(0.7)
    section.footer_distance = Cm(0.7)

    content_width_cm = (section.page_width.cm - section.left_margin.cm - section.right_margin.cm)

    normal = doc.styles["Normal"]
    normal.font.name = "宋体"
    normal.font.size = Pt(10.5)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.paragraph_format.line_spacing = 1.3 if portrait else 1.25
    normal.paragraph_format.space_after = Pt(5 if portrait else 4)

    heading_specs = {
        "Heading 1": (16, (31, 78, 121)),
        "Heading 2": (13, (31, 78, 121)),
        "Heading 3": (11.5, (50, 50, 50)),
    }
    for style_name, (size, color) in heading_specs.items():
        style = doc.styles[style_name]
        style.font.name = "微软雅黑"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(*color)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
        style.paragraph_format.space_before = Pt(12 if style_name != "Heading 1" else 6)
        style.paragraph_format.space_after = Pt(6)
        style.paragraph_format.keep_with_next = True

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = header.add_run(header_text)
    set_run_font(run, font_name="微软雅黑", size=8.5, color=(100, 100, 100))
    add_page_number(section.footer.paragraphs[0])

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        if stripped.startswith("```"):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            add_code_block(doc, code_lines)
            spacer = doc.add_paragraph()
            spacer.paragraph_format.space_after = Pt(2)
            continue

        if stripped.startswith("|") and i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            add_table(doc, parse_table(table_lines), content_width_cm)
            doc.add_paragraph().paragraph_format.space_after = Pt(2)
            continue

        if stripped == "---":
            i += 1
            continue

        if stripped.startswith("# "):
            p = doc.add_paragraph(style="Heading 1")
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_inline(p, stripped[2:], size=18 if not portrait else 20,
                       base_bold=True, color=(31, 78, 121))
        elif stripped.startswith("## "):
            p = doc.add_paragraph(style="Heading 2")
            add_inline(p, stripped[3:], size=13, base_bold=True, color=(31, 78, 121))
        elif stripped.startswith("### "):
            p = doc.add_paragraph(style="Heading 3")
            add_inline(p, stripped[4:], size=11.5, base_bold=True, color=(50, 50, 50))
        elif stripped.startswith(">"):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.5)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(6)
            add_inline(p, stripped.lstrip("> ").strip(), size=10, color=(90, 90, 90))
        elif re.match(r"^\s*[-*]\s+", line):
            content = re.sub(r"^\s*[-*]\s+", "", line)
            p = doc.add_paragraph(style="List Bullet")
            add_inline(p, content, size=10.5)
        elif re.match(r"^\s*\d+\.\s+", line):
            content = re.sub(r"^\s*\d+\.\s+", "", line)
            p = doc.add_paragraph(style="List Number")
            add_inline(p, content, size=10.5)
        else:
            p = doc.add_paragraph()
            add_inline(p, stripped, size=10.5)
        i += 1

    zoom = doc.settings.element.find(qn("w:zoom"))
    if zoom is None:
        zoom = OxmlElement("w:zoom")
        doc.settings.element.append(zoom)
    zoom.set(qn("w:percent"), "100")

    props = doc.core_properties
    props.title = title
    props.subject = subtitle or title
    props.author = "项目工作文档"
    doc.save(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--header", default=DEFAULT_HEADER)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--subtitle", default="")
    parser.add_argument("--portrait", action="store_true")
    args = parser.parse_args()
    build(args.source, args.output, header_text=args.header, title=args.title,
          subtitle=args.subtitle, portrait=args.portrait)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
