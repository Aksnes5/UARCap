"""Build the complete UARCap v0.8 submission package (DOCX, CSV tables, BibTeX, LaTeX, Cover Letter).

Reads directly from:
  - paper/UARCap_paper_v0.8.md
  - paper/UARCap_Supplementary_Materials.md
  - paper/REFERENCES_VERIFIED.bib
  - paper/v0.8_assets/fig*.{pdf,svg,png}
Produces:
  - submission_v0.8/
"""

import csv
import re
import shutil
from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
ASSETS = PAPER / "v0.8_assets"
SUB = ROOT / "submission_v0.8"
FIG_OUT = SUB / "figures"
TAB_OUT = SUB / "tables"
TEX_OUT = SUB / "latex"
TEX_FIG = TEX_OUT / "figures"

NAVY_RGB = RGBColor(0x1B, 0x4F, 0x72)
INK_RGB = RGBColor(0x1A, 0x1A, 0x1A)
MUTED_RGB = RGBColor(0x55, 0x62, 0x70)

CITED_BIB_KEYS = [
    "Anderson2016SPICE",
    "Banerjee2005METEOR",
    "Chen2011MSVD",
    "Cuconasu2024PowerNoise",
    "Dai2025SePer",
    "Fang2026EVCap",
    "Jeon2025Sali4Vid",
    "Jeong2025VideoRAG",
    "Joren2025SufficientContext",
    "Kim2024CM2",
    "Kim2025HiCM2",
    "Li2023BLIP2",
    "Lin2004ROUGE",
    "Liu2024LostMiddle",
    "LMSYS2023Vicuna13",
    "Ni2022XCLIP",
    "Papineni2002BLEU",
    "Ramos2023SmallCap",
    "Ren2026VideoRAG",
    "Sarto2022RATransformer",
    "Shao2024MassiveDS",
    "Sun2024SURf",
    "Taguchi2025AdaptiveK",
    "Vedantam2015CIDEr",
    "VideoUFOCard",
    "Wang2025VideoUFO",
    "Xu2024EgoInstructor",
    "Xu2016MSRVTT",
    "Zhang2023VideoLLaMA",
    "Zhang2021OpenBook",
    "Zhao2026MADRAG",
]


def ensure_dirs():
    for d in (SUB, FIG_OUT, TAB_OUT, TEX_OUT, TEX_FIG):
        d.mkdir(parents=True, exist_ok=True)


def build_verified_bib():
    raw = (PAPER / "REFERENCES_VERIFIED.bib").read_text(encoding="utf-8")
    # Split entries cleanly by top-level '@'
    blocks = re.split(r"(?m)(?=^@\w+\{)", raw)
    entries = {}
    for b in blocks:
        m = re.match(r"^@\w+\{([^,]+),", b.strip())
        if m:
            entries[m.group(1).strip()] = b.strip()
    missing = [k for k in CITED_BIB_KEYS if k not in entries]
    if missing:
        raise ValueError(f"Missing BibTeX keys in REFERENCES_VERIFIED.bib: {missing}")

    header = (
        "% Verified bibliography for UARCap v0.8 submission package.\n"
        "% Subset of paper/REFERENCES_VERIFIED.bib containing exactly the 31 cited references.\n\n"
    )
    bib_text = header + "\n\n".join(entries[k] for k in CITED_BIB_KEYS) + "\n"
    (SUB / "UARCap_References.bib").write_text(bib_text, encoding="utf-8")
    (TEX_OUT / "UARCap_References.bib").write_text(bib_text, encoding="utf-8")
    print(f"Wrote {len(CITED_BIB_KEYS)} verified BibTeX entries.")


def copy_figures_and_markdown():
    for f in ASSETS.iterdir():
        if f.is_file() and (f.name.startswith("fig") or f.name == "build_figures_v08.py"):
            shutil.copy2(f, FIG_OUT / f.name)
            if f.suffix == ".pdf":
                shutil.copy2(f, TEX_FIG / f.name)

    ms_text = (PAPER / "UARCap_paper_v0.8.md").read_text(encoding="utf-8")
    ms_sub = ms_text.replace("v0.8_assets/", "figures/")
    (SUB / "UARCap_Manuscript_v0.8.md").write_text(ms_sub, encoding="utf-8")
    shutil.copy2(PAPER / "UARCap_Supplementary_Materials.md", SUB / "UARCap_Supplementary_Materials.md")
    print("Copied figures and synchronized Markdown files.")


def strip_md_inline(s: str) -> str:
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = s.replace("$k$", "k").replace("$d_z$", "d_z").replace("$n$", "n")
    s = s.replace("$\\rho$", "rho").replace("$p$", "p").replace("$q$", "q")
    s = s.replace("$1 \\times 32 \\times 768$", "1 x 32 x 768")
    s = re.sub(r"<sup>([^<]+)</sup>", r"^\1", s)
    return s.strip()


def extract_markdown_tables(md_text: str):
    lines = md_text.splitlines()
    tables = []
    i = 0
    while i < len(lines):
        m = re.match(r"^\*\*(Table [A-Z]?\d+)\.\*\*\s*(.+)$", lines[i].strip())
        if m:
            tid, caption = m.group(1), m.group(2)
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("|"):
                j += 1
            rows = []
            while j < len(lines) and lines[j].strip().startswith("|"):
                parts = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                if not all(re.match(r"^:?-+:?$", p) for p in parts):
                    rows.append(parts)
                j += 1
            tables.append((tid, caption, rows))
            i = j
        else:
            i += 1
    return tables


def export_tables():
    ms_tables = extract_markdown_tables((PAPER / "UARCap_paper_v0.8.md").read_text(encoding="utf-8"))
    sup_tables = extract_markdown_tables((PAPER / "UARCap_Supplementary_Materials.md").read_text(encoding="utf-8"))
    all_md = ["# Exported Manuscript and Supplementary Tables (UARCap v0.8)\n"]
    slug_map = {
        "Table 1": "Table1_scale_grid",
        "Table 2": "Table2_test_means",
        "Table 3": "Table3_test_inference",
        "Table 4": "Table4_followup_means",
        "Table 5": "Table5_five_test_holm",
        "Table 6": "Table6_oracle",
        "Table S1": "TableS1_corpus_manifests",
        "Table S2": "TableS2_full_benchmark_contrasts",
        "Table S3": "TableS3_repaired_controls",
        "Table S4": "TableS4_experiment_b_matching",
        "Table S5": "TableS5_followup_full_stats",
        "Table S6": "TableS6_runtime_integrity",
        "Table S7": "TableS7_output_behavior",
    }
    for tid, cap, rows in ms_tables + sup_tables:
        fname = slug_map.get(tid, tid.replace(" ", "")) + ".csv"
        with (TAB_OUT / fname).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            for r in rows:
                w.writerow([strip_md_inline(c) for c in r])
        all_md.append(f"## {tid}\n\n**{tid}.** {cap}\n")
        if rows:
            all_md.append("| " + " | ".join(rows[0]) + " |")
            all_md.append("| " + " | ".join("---" for _ in rows[0]) + " |")
            for r in rows[1:]:
                all_md.append("| " + " | ".join(r) + " |")
        all_md.append("")
    (TAB_OUT / "ALL_TABLES.md").write_text("\n".join(all_md), encoding="utf-8")
    print(f"Exported {len(ms_tables)} main tables and {len(sup_tables)} supplementary tables.")


# ---------------- DOCX FORMATTING HELPERS ----------------

def set_cell_margins(cell, top=55, bottom=55, left=80, right=80):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement("w:tcMar")
    for m, val in (("top", top), ("bottom", bottom), ("left", left), ("right", right)):
        node = OxmlElement(f"w:{m}")
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")
        tcMar.append(node)
    tcPr.append(tcMar)


def set_cell_borders(cell, top=None, bottom=None):
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for side in ("top", "left", "bottom", "right"):
        b = OxmlElement(f"w:{side}")
        spec = top if side == "top" else bottom if side == "bottom" else None
        if spec:
            b.set(qn("w:val"), "single")
            b.set(qn("w:sz"), str(spec[0]))
            b.set(qn("w:space"), "0")
            b.set(qn("w:color"), spec[1])
        else:
            b.set(qn("w:val"), "none")
        tcBorders.append(b)
    tcPr.append(tcBorders)


def set_row_cant_split(row):
    trPr = row._tr.get_or_add_trPr()
    trPr.append(OxmlElement("w:cantSplit"))


def set_row_header(row):
    trPr = row._tr.get_or_add_trPr()
    trPr.append(OxmlElement("w:tblHeader"))


def set_cell_shading(cell, hex_color: str):
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>'))


def add_page_number_footer(section):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.font.name = "Times New Roman"
    run.font.size = Pt(9.5)
    run.font.color.rgb = MUTED_RGB
    fld1 = OxmlElement("w:fldChar")
    fld1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld2 = OxmlElement("w:fldChar")
    fld2.set(qn("w:fldCharType"), "end")
    run._r.append(fld1)
    run._r.append(instr)
    run._r.append(fld2)


def add_running_header(section, text: str):
    header = section.header
    p = header.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(8.5)
    r.font.italic = True
    r.font.color.rgb = MUTED_RGB


def normalize_math_tex(expr: str) -> str:
    """Convert LaTeX math tokens to Unicode while preserving _{...} and ^{...} markers."""
    rep = [
        (r"\widehat{y}", "ŷ"),
        (r"\tilde{k}", "k̃"),
        (r"\big(", "("),
        (r"\big)", ")"),
        (r"\mathrm{ROUGE\text{-}L}", "ROUGE-L"),
        (r"\mathrm{df}", "df"),
        (r"\log", "log"),
        (r"\frac{1}{19}", "1/19"),
        (r"\sum", "Σ"),
        (r"\{", "{"),
        (r"\}", "}"),
        (r"{,}", ","),
        (r"\in", "∈"),
        (r"\ne", " ≠ "),
        (r"\neq", " ≠ "),
        (r"\le", "≤"),
        (r"\ge", "≥"),
        (r"\times", "×"),
        (r"\approx", "≈"),
        (r"\to", "→"),
        (r"\pm", "±"),
        (r"\rho", "ρ"),
        (r"\alpha", "α"),
        (r"\theta", "θ"),
        (r"\,", ","),
    ]
    out = expr
    for a, b in rep:
        out = out.replace(a, b)
    out = re.sub(r"\s+", " ", out).strip()
    out = out.replace("Σ_", "Σ_")
    return out


def add_math_runs(paragraph, expr: str, base_size=10.5, base_color=INK_RGB, base_bold=False):
    """Render normalized math expression with true Word subscripts/superscripts."""
    s = normalize_math_tex(expr)
    # Match _{...}, _X, ^{...}, ^X (including Greek letters and digits)
    tok_pat = re.compile(r"(_\{[^}]+\}|_[A-Za-z0-9θρα]+|\^\{[^}]+\}|\^[A-Za-z0-9\-+]+)")
    pos = 0

    def emit_math_text(chunk: str, sub=False, sup=False):
        if not chunk:
            return
        # Keep upright tokens (ROUGE-L, df, log, digits, operators) upright and variables italic
        up_pat = re.compile(r"(ROUGE-L|\blog\b|\bdf\b|[0-9.,/()\[\]{}|=+\-−×∈≠≤≥≈→±Σ ]+)")
        upos = 0
        for um in up_pat.finditer(chunk):
            if um.start() > upos:
                r = paragraph.add_run(chunk[upos:um.start()])
                r.font.name = "Cambria Math"
                r.font.size = Pt(base_size - 1.5 if (sub or sup) else base_size)
                r.font.color.rgb = base_color
                r.bold = base_bold
                r.italic = True
                r.font.subscript = sub
                r.font.superscript = sup
            r = paragraph.add_run(um.group(0))
            r.font.name = "Cambria Math"
            r.font.size = Pt(base_size - 1.5 if (sub or sup) else base_size)
            r.font.color.rgb = base_color
            r.bold = base_bold
            r.italic = False
            r.font.subscript = sub
            r.font.superscript = sup
            upos = um.end()
        if upos < len(chunk):
            r = paragraph.add_run(chunk[upos:])
            r.font.name = "Cambria Math"
            r.font.size = Pt(base_size - 1.5 if (sub or sup) else base_size)
            r.font.color.rgb = base_color
            r.bold = base_bold
            r.italic = True
            r.font.subscript = sub
            r.font.superscript = sup

    for m in tok_pat.finditer(s):
        if m.start() > pos:
            emit_math_text(s[pos:m.start()])
        tok = m.group(0)
        is_sub = tok.startswith("_")
        inner = tok[2:-1] if tok[1] == "{" else tok[1:]
        emit_math_text(inner.strip(), sub=is_sub, sup=(not is_sub))
        pos = m.end()
    if pos < len(s):
        emit_math_text(s[pos:])


def add_rich_runs(
    paragraph,
    text: str,
    base_size=10.5,
    base_color=INK_RGB,
    base_bold=False,
    base_italic=False,
    base_sup=False,
    base_sub=False,
):
    """Recursive inline Markdown + math + <sup>/<sub> renderer for python-docx."""
    # Priority: <sup>, <sub>, **bold** (allowing nested *, $, <sup>), $math$, `code`, *italic*
    pattern = re.compile(
        r"(<sup>.*?</sup>|<sub>.*?</sub>|\*\*(?!\s).+?(?<!\s)\*\*|\$[^$\n]+\$|`[^`\n]+`|\*(?!\s)[^*\n]+?(?<!\s)\*)"
    )

    def emit_plain(chunk: str):
        if not chunk:
            return
        r = paragraph.add_run(chunk)
        r.font.name = "Times New Roman"
        r.font.size = Pt(base_size)
        r.font.color.rgb = base_color
        r.bold = base_bold
        r.italic = base_italic
        if base_sup:
            r.font.superscript = True
        if base_sub:
            r.font.subscript = True

    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            emit_plain(text[pos:m.start()])
        tok = m.group(0)
        if tok.startswith("<sup>"):
            add_rich_runs(
                paragraph, tok[5:-6],
                base_size=base_size, base_color=base_color,
                base_bold=base_bold, base_italic=base_italic,
                base_sup=True, base_sub=base_sub,
            )
        elif tok.startswith("<sub>"):
            add_rich_runs(
                paragraph, tok[5:-6],
                base_size=base_size, base_color=base_color,
                base_bold=base_bold, base_italic=base_italic,
                base_sup=base_sup, base_sub=True,
            )
        elif tok.startswith("**"):
            add_rich_runs(
                paragraph, tok[2:-2],
                base_size=base_size, base_color=base_color,
                base_bold=True, base_italic=base_italic,
                base_sup=base_sup, base_sub=base_sub,
            )
        elif tok.startswith("$"):
            add_math_runs(
                paragraph, tok[1:-1],
                base_size=base_size, base_color=base_color,
                base_bold=base_bold,
            )
        elif tok.startswith("`"):
            code_txt = tok[1:-1]
            # Allow clean line wrapping inside long SHA-256 hex strings
            if re.fullmatch(r"[0-9a-fA-F]{40,}", code_txt):
                code_txt = "\u200b".join(code_txt[k:k+16] for k in range(0, len(code_txt), 16))
            r = paragraph.add_run(code_txt)
            r.font.name = "Consolas"
            r.font.size = Pt(base_size - 1.0)
            r.font.color.rgb = base_color
            r.bold = base_bold
        elif tok.startswith("*"):
            add_rich_runs(
                paragraph, tok[1:-1],
                base_size=base_size, base_color=base_color,
                base_bold=base_bold, base_italic=True,
                base_sup=base_sup, base_sub=base_sub,
            )
        pos = m.end()
    if pos < len(text):
        emit_plain(text[pos:])


def add_booktabs_table(doc, rows, font_size=8.8):
    if not rows:
        return
    ncols = len(rows[0])
    tbl = doc.add_table(rows=len(rows), cols=ncols)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = True

    # Set table width to 100% of printable area
    tblPr = tbl._tbl.tblPr
    tblW = OxmlElement("w:tblW")
    tblW.set(qn("w:w"), "5000")
    tblW.set(qn("w:type"), "pct")
    tblPr.append(tblW)

    keep_together = (len(rows) <= 10)  # Keep compact tables intact on one page

    for r_idx, row_data in enumerate(rows):
        row = tbl.rows[r_idx]
        set_row_cant_split(row)
        if r_idx == 0:
            set_row_header(row)
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=46, bottom=46, left=62, right=62)
            if r_idx == 0:
                set_cell_borders(cell, top=(12, "1B4F72"), bottom=(6, "1B4F72"))
                set_cell_shading(cell, "EFF4F7")
            elif r_idx == len(rows) - 1:
                set_cell_borders(cell, bottom=(12, "1B4F72"))
            else:
                set_cell_borders(cell, bottom=(2, "E5EAEE"))
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.08
            if keep_together and r_idx < len(rows) - 1:
                p.paragraph_format.keep_with_next = True
            # Right-align numeric cells except first column
            clean = strip_md_inline(val)
            is_num = c_idx > 0 and bool(re.match(r"^[+\-−]?\d|^\.\d|^\[|^—$", clean))
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if is_num else WD_ALIGN_PARAGRAPH.LEFT
            add_rich_runs(p, val, base_size=font_size, base_bold=(r_idx == 0))

    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(0)
    sp.paragraph_format.space_after = Pt(6)


def md_to_docx(md_path: Path, docx_path: Path, running_header: str, fig_base: Path, is_supp=False):
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Inches(8.27)   # A4 width
    sec.page_height = Inches(11.69) # A4 height
    margin = Inches(0.85 if is_supp else 0.95)
    sec.top_margin = margin
    sec.bottom_margin = margin
    sec.left_margin = margin
    sec.right_margin = margin
    sec.different_first_page_header_footer = True
    add_running_header(sec, running_header)
    add_page_number_footer(sec)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = INK_RGB

    text = md_path.read_text(encoding="utf-8")
    # Strip YAML frontmatter
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            text = parts[2]

    lines = text.splitlines()
    i = 0
    in_references = False
    eq_num = 0
    while i < len(lines):
        line = lines[i].rstrip()
        s = line.strip()

        if not s or s == "---":
            i += 1
            continue

        # H1 Title
        if s.startswith("# "):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(8)
            p.paragraph_format.line_spacing = 1.15
            add_rich_runs(p, s[2:].strip(), base_size=16.0, base_color=NAVY_RGB, base_bold=True)
            i += 1
            continue

        # H2 Section
        if s.startswith("## "):
            heading = s[3:].strip()
            in_references = (heading.lower() == "references")
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(13)
            p.paragraph_format.space_after = Pt(5)
            p.paragraph_format.keep_with_next = True
            add_rich_runs(p, heading, base_size=12.5, base_color=NAVY_RGB, base_bold=True)
            # Subtle bottom rule under major section headings
            pBdr = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="4" w:space="2" w:color="C5D4DE"/></w:pBdr>')
            p._p.get_or_add_pPr().append(pBdr)
            i += 1
            continue

        # H3 Subsection
        if s.startswith("### "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(3.5)
            p.paragraph_format.keep_with_next = True
            add_rich_runs(p, s[4:].strip(), base_size=11.0, base_color=INK_RGB, base_bold=True)
            i += 1
            continue

        # Display equation $$ ... $$
        if s == "$$":
            eq_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() != "$$":
                eq_lines.append(lines[i].strip())
                i += 1
            i += 1  # skip closing $$
            eq_num += 1
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
            add_math_runs(p, " ".join(eq_lines), base_size=10.5, base_color=INK_RGB)
            r_num = p.add_run(f"        ({eq_num})")
            r_num.font.name = "Times New Roman"
            r_num.font.size = Pt(10.5)
            r_num.font.color.rgb = INK_RGB
            continue

        # Image ![alt](path)
        m_img = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", s)
        if m_img:
            img_rel = m_img.group(2)
            img_file = fig_base / Path(img_rel).name
            if not img_file.exists():
                raise FileNotFoundError(f"Figure not found: {img_file}")
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.keep_with_next = True
            run = p.add_run()
            run.add_picture(str(img_file), width=Inches(6.25))
            i += 1
            continue

        # Figure caption
        if re.match(r"^\*\*Figure \d+\.\*\*", s):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(9)
            p.paragraph_format.line_spacing = 1.12
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            add_rich_runs(p, s, base_size=9.5, base_color=INK_RGB)
            i += 1
            continue

        # Table caption
        if re.match(r"^\*\*Table [A-Z]?\d+\.\*\*", s):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.12
            p.paragraph_format.keep_with_next = True
            add_rich_runs(p, s, base_size=9.5, base_color=INK_RGB)
            i += 1
            continue

        # Markdown table
        if s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                parts = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.match(r"^:?-+:?$", p) for p in parts):
                    rows.append(parts)
                i += 1
            # Compact font for wider tables (e.g. 9-col tables)
            ncols = len(rows[0]) if rows else 4
            fsize = 7.8 if ncols >= 9 else 8.4 if ncols >= 7 else 9.0
            add_booktabs_table(doc, rows, font_size=fsize)
            continue

        # Bullet list
        if s.startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.space_before = Pt(1.5)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.14
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            add_rich_runs(p, s[2:].strip(), base_size=10.3)
            i += 1
            continue

        # Numbered list
        m_num = re.match(r"^(\d+)\.\s+(.+)$", s)
        if m_num:
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.28)
            p.paragraph_format.first_line_indent = Inches(-0.20)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(3.5)
            p.paragraph_format.line_spacing = 1.14
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            add_rich_runs(p, f"**{m_num.group(1)}.** {m_num.group(2)}", base_size=10.3)
            i += 1
            continue

        # Normal paragraph (or reference hanging indent)
        p = doc.add_paragraph()
        if in_references:
            p.paragraph_format.left_indent = Inches(0.32)
            p.paragraph_format.first_line_indent = Inches(-0.32)
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(3.5)
            p.paragraph_format.line_spacing = 1.10
            add_rich_runs(p, s, base_size=9.2)
        else:
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(5.5)
            p.paragraph_format.line_spacing = 1.16
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            add_rich_runs(p, s, base_size=10.5)
        i += 1

    doc.save(str(docx_path))
    print(f"Built DOCX: {docx_path.name}")


def build_cover_letter():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Inches(8.27)
    sec.page_height = Inches(11.69)
    sec.top_margin = Inches(0.80)
    sec.bottom_margin = Inches(0.80)
    sec.left_margin = Inches(0.95)
    sec.right_margin = Inches(0.95)

    def para(text, space_after=6, bold=False, italic=False, size=10.2, color=INK_RGB):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.12
        add_rich_runs(p, text, base_size=size, base_color=color, base_bold=bold, base_italic=italic)
        return p

    para("[Date: YYYY-MM-DD — TO BE COMPLETED BY AUTHORS]", space_after=10, color=MUTED_RGB)
    para("**To:** [Editor-in-Chief Name / Editorial Office]", space_after=2)
    para("**Journal:** [Target Journal Name]", space_after=10)
    para("Dear Editor-in-Chief,", space_after=8)
    para(
        'We are pleased to submit our original research manuscript entitled **"When Does External Retrieval Help Video Captioning? Evidence Budgets in a Million-Clip Memory"** for consideration for publication in *[Target Journal Name]*.'
    )
    para(
        "Retrieval-augmented video captioning commonly assumes that expanding an external memory or supplying more retrieved examples improves generated descriptions. In this empirical diagnostic study, we hold a fixed multimodal video-captioning architecture constant and jointly examine six nested VideoUFO external memories (10,000 to 1,091,712 clips), cross-benchmark joint evidence budgets on MSR-VTT and MSVD, and two prespecified controlled interventions evaluated with source-cluster bootstrap, sign-flip permutation, and Holm multiplicity adjustment."
    )
    para(
        "Our study makes three distinct contributions beyond prior in-domain architecture papers, including Fang et al. (*Displays*, 2026), whose frozen captioning interface we inherit as a controlled testbed:"
    )
    for b in (
        "**Access–utility dissociation at million-clip scale:** Expanding the external memory substantially increases selected X-CLIP similarity and replaces nearly every top-one candidate without producing consistent fixed-budget gains in caption quality, while smaller joint budgets consistently outperform seven examples on both MSR-VTT and MSVD.",
        "**Matched zero-evidence net utility (Experiment A):** Retaining the complete prompt scaffold and cross-gating fusion pathway at zero external input (n = 2,690 MSR-VTT queries; 2,004 video sources), one external caption–clip pair confers a statistically supported net gain of +1.40 CIDEr points that survives five-test Holm adjustment (p = .0165), whereas seven pairs drop −1.54 points below zero evidence (passing within-family Holm p = .0271, though not five-test Holm p = .0813).",
        "**Visual-tensor-verified textual attribution boundaries (Experiment B):** Verifying exact per-query identity of all three retrieved visual tensors and intermediate fused representations (n = 2,618; 1,969 sources), appending four length- and overlap-matched higher- or lower-similarity captions yields directional decreases (−1.27 and −1.13 points) that do not pass B-family Holm adjustment, with an inconclusive contrast between similarity bands (−0.14, p = .7698). We report these evidential boundaries transparently alongside repaired validation diagnostics and full offline verification."
    ):
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.11
        add_rich_runs(p, b, base_size=9.9)

    para(
        "We confirm that this manuscript is original, has not been published previously, and is not under consideration elsewhere. Because this work builds directly upon the frozen captioning interface of Fang et al. (*Displays* 93, 2026, 103420), we explicitly distinguish our external-memory diagnostic design, matched zero-evidence control, fixed-visual text intervention, and full-caption consensus reranking approximation from the prior architectural study. [Authors to confirm shared-authorship and competing-interest wording per target journal policy.]"
    )
    para(
        "All prespecified protocols, frozen evaluation outputs, integrity receipts, source-cluster statistical tables, and figure-generation code are preserved in a reproducible archive. Thank you for considering our manuscript. We look forward to your editorial evaluation.",
        space_after=10
    )
    para("Sincerely,", space_after=12)
    para("**[Corresponding Author Name]**", space_after=1.5)
    para("[Title / Position], [Department]", space_after=1.5)
    para("[Institution Name, City, Country]", space_after=1.5)
    para("Email: [corresponding.author@institution.edu]", space_after=1.5)
    para("ORCID: [0000-0000-0000-0000]", space_after=1.5)

    out = SUB / "UARCap_Cover_Letter_Draft.docx"
    doc.save(str(out))
    print(f"Built Cover Letter: {out.name}")


# ---------------- LATEX GENERATION ----------------

CITATION_PATTERNS = [
    (r"\(Fang et al\. 2026, Section 3\.4\)", r"\\citep[Section~3.4]{Fang2026EVCap}"),
    (r"\(Fang et al\. 2026, Table 1\)", r"\\citep[Table~1]{Fang2026EVCap}"),
    (r"\(Fang et al\. 2026\)", r"\\citep{Fang2026EVCap}"),
    (r"Fang et al\. \(2026, Section 3\.4\)", r"\\citet[Section~3.4]{Fang2026EVCap}"),
    (r"Fang et al\. \(2026, Table 1\)", r"\\citet[Table~1]{Fang2026EVCap}"),
    (r"Fang et al\. \(2026\)", r"\\citet{Fang2026EVCap}"),
    (r"\(Zhang et al\. 2021\)", r"\\citep{Zhang2021OpenBook}"),
    (r"\(Xu et al\. 2024\)", r"\\citep{Xu2024EgoInstructor}"),
    (r"\(Kim et al\. 2024\)", r"\\citep{Kim2024CM2}"),
    (r"\(Kim et al\. 2025\)", r"\\citep{Kim2025HiCM2}"),
    (r"\(Jeon et al\. 2025\)", r"\\citep{Jeon2025Sali4Vid}"),
    (r"\(Ramos et al\. 2023\)", r"\\citep{Ramos2023SmallCap}"),
    (r"\(Sun et al\. 2024\)", r"\\citep{Sun2024SURf}"),
    (r"\(Jeong et al\. 2025; Ren et al\. 2026\)", r"\\citep{Jeong2025VideoRAG,Ren2026VideoRAG}"),
    (r"\(Sarto et al\. 2022\)", r"\\citep{Sarto2022RATransformer}"),
    (r"\(Dai et al\. 2025\)", r"\\citep{Dai2025SePer}"),
    (r"\(Joren et al\. 2025\)", r"\\citep{Joren2025SufficientContext}"),
    (r"\(Shao et al\. 2024\)", r"\\citep{Shao2024MassiveDS}"),
    (r"\(Liu et al\. 2024\)", r"\\citep{Liu2024LostMiddle}"),
    (r"\(Cuconasu et al\. 2024\)", r"\\citep{Cuconasu2024PowerNoise}"),
    (r"\(Zhao et al\. 2026\)", r"\\citep{Zhao2026MADRAG}"),
    (r"\(Taguchi et al\. 2025\)", r"\\citep{Taguchi2025AdaptiveK}"),
    (r"\(Zhang et al\. 2023\)", r"\\citep{Zhang2023VideoLLaMA}"),
    (r"\(Li et al\. 2023\)", r"\\citep{Li2023BLIP2}"),
    (r"\(LMSYS 2023\)", r"\\citep{LMSYS2023Vicuna13}"),
    (r"\(Wang and Yang 2025; VideoUFO 2025\)", r"\\citep{Wang2025VideoUFO,VideoUFOCard}"),
    (r"\(VideoUFO 2025\)", r"\\citep{VideoUFOCard}"),
    (r"\(Ni et al\. 2022\)", r"\\citep{Ni2022XCLIP}"),
    (r"\(Xu et al\. 2016\)", r"\\citep{Xu2016MSRVTT}"),
    (r"\(Chen and Dolan 2011; Fang et al\. 2026\)", r"\\citep{Chen2011MSVD,Fang2026EVCap}"),
    (r"\(Chen and Dolan 2011\)", r"\\citep{Chen2011MSVD}"),
    (r"\(Vedantam et al\. 2015\)", r"\\citep{Vedantam2015CIDEr}"),
    (r"\(Papineni et al\. 2002; Banerjee and Lavie 2005; Lin 2004; Anderson et al\. 2016\)",
     r"\\citep{Papineni2002BLEU,Banerjee2005METEOR,Lin2004ROUGE,Anderson2016SPICE}"),
]


def tex_escape_inline(s: str) -> str:
    # Protect inline $...$ math segments first
    math_slots = []
    def save_math(m):
        math_slots.append(m.group(0))
        return f"@@MATH{len(math_slots)-1}@@"

    s = re.sub(r"\$[^$\n]+\$", save_math, s)

    # Apply citations before escaping
    cite_slots = []
    for pat, repl in CITATION_PATTERNS:
        def save_cite(m, r=repl):
            cite_slots.append(r)
            return f"@@CITE{len(cite_slots)-1}@@"
        s = re.sub(pat, save_cite, s)

    def esc_plain(t: str) -> str:
        t = t.replace("\\", r"\textbackslash{}")
        for ch, rep in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")):
            t = t.replace(ch, rep)
        t = t.replace("−", "$-$").replace("±", r"$\pm$").replace("×", r"$\times$")
        t = t.replace("→", r"$\rightarrow$").replace("—", "---").replace("–", "--")
        t = t.replace("“", "``").replace("”", "''").replace("’", "'").replace("²", r"$^2$")
        return t

    # Recursive inline markdown converter supporting nested **... *italic* ...** and **@@MATH@@**
    tok_pat = re.compile(
        r"(<sup>.*?</sup>|\*\*(?!\s).+?(?<!\s)\*\*|`[^`\n]+`|\*(?!\s)[^*\n]+?(?<!\s)\*)"
    )

    def convert_md(chunk: str) -> str:
        out_parts = []
        pos = 0
        for m in tok_pat.finditer(chunk):
            if m.start() > pos:
                out_parts.append(esc_plain(chunk[pos:m.start()]))
            tok = m.group(0)
            if tok.startswith("<sup>"):
                out_parts.append(r"\textsuperscript{" + esc_plain(tok[5:-6]) + "}")
            elif tok.startswith("**"):
                out_parts.append(r"\textbf{" + convert_md(tok[2:-2]) + "}")
            elif tok.startswith("`"):
                out_parts.append(r"\texttt{" + esc_plain(tok[1:-1]) + "}")
            elif tok.startswith("*"):
                out_parts.append(r"\emph{" + convert_md(tok[1:-1]) + "}")
            pos = m.end()
        if pos < len(chunk):
            out_parts.append(esc_plain(chunk[pos:]))
        return "".join(out_parts)

    s = convert_md(s)

    for idx, c in enumerate(cite_slots):
        s = s.replace(f"@@CITE{idx}@@", c)
    for idx, m in enumerate(math_slots):
        s = s.replace(f"@@MATH{idx}@@", m)
    return s


def md_to_latex(md_path: Path, tex_path: Path, is_supp=False):
    text = md_path.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            text = parts[2]

    lines = text.splitlines()
    out = [
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage[margin=2.4cm]{geometry}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{mathptmx}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{xcolor}",
        r"\usepackage[round,authoryear]{natbib}",
        r"\usepackage[colorlinks=true,citecolor=blue!50!black,linkcolor=blue!50!black,urlcolor=blue!50!black]{hyperref}",
        r"\usepackage{caption}",
        r"\captionsetup{font=small,labelfont=bf}",
        r"",
    ]

    if is_supp:
        out += [
            r"\title{\textbf{Supplementary Materials}\\[4pt]\large When Does External Retrieval Help Video Captioning? Evidence Budgets in a Million-Clip Memory}",
            r"\author{[Author List --- TO BE COMPLETED BY AUTHORS]}",
            r"\date{}",
            r"\renewcommand{\thetable}{S\arabic{table}}",
            r"\begin{document}",
            r"\maketitle",
            r"",
        ]
    else:
        out += [
            r"\title{\textbf{When Does External Retrieval Help Video Captioning? Evidence Budgets in a Million-Clip Memory}}",
            r"\author{[Author Name 1]$^{1}$, [Author Name 2]$^{1}$, \ldots, [Corresponding Author Name]$^{1,*}$\\[4pt]",
            r"\small $^{1}$[Department, Institution, City, Country --- TO BE COMPLETED BY AUTHORS]\\",
            r"\small $^{*}$Corresponding author: \texttt{[email@institution.edu --- TO BE COMPLETED BY AUTHORS]}}",
            r"\date{}",
            r"\begin{document}",
            r"\maketitle",
            r"",
        ]

    i = 0
    in_itemize = False
    in_enum = False
    skip_refs = False
    pending_table_cap = None
    pending_table_label = None

    def close_lists():
        nonlocal in_itemize, in_enum
        if in_itemize:
            out.append(r"\end{itemize}")
            in_itemize = False
        if in_enum:
            out.append(r"\end{enumerate}")
            in_enum = False

    while i < len(lines):
        s = lines[i].strip()
        if not s or s == "---" or s.startswith("# ") or s.startswith("**[Author") or s.startswith("<sup>") or s.startswith("**Manuscript:**") or s.startswith("**Authors:**"):
            i += 1
            continue

        if s.startswith("## "):
            close_lists()
            h = s[3:].strip()
            if h.lower() == "abstract":
                i += 1
                abs_lines = []
                while i < len(lines) and not lines[i].strip().startswith("**Keywords:") and not lines[i].strip().startswith("## "):
                    if lines[i].strip():
                        abs_lines.append(lines[i].strip())
                    i += 1
                out.append(r"\begin{abstract}")
                out.append(tex_escape_inline(" ".join(abs_lines)))
                out.append(r"\end{abstract}")
                continue
            if h.lower() == "references":
                skip_refs = True
                break
            if h.lower() == "declarations":
                out.append(r"\section*{Declarations}")
                i += 1
                continue
            clean_h = re.sub(r"^(?:\d+|S\d+)\.\s*", "", h)
            out.append(f"\\section{{{tex_escape_inline(clean_h)}}}")
            i += 1
            continue

        if s.startswith("### "):
            close_lists()
            clean_h = re.sub(r"^\d+\.\d+\s*", "", s[4:].strip())
            out.append(f"\\subsection{{{tex_escape_inline(clean_h)}}}")
            i += 1
            continue

        if s.startswith("**Keywords:**"):
            kw = s.replace("**Keywords:**", "").strip()
            out.append(f"\\noindent\\textbf{{Keywords:}} {tex_escape_inline(kw)}\n")
            i += 1
            continue

        if s == "$$":
            close_lists()
            eq = []
            i += 1
            while i < len(lines) and lines[i].strip() != "$$":
                eq.append(lines[i].strip())
                i += 1
            i += 1
            out.append(r"\begin{equation}")
            out.append(" ".join(eq))
            out.append(r"\end{equation}")
            continue

        m_img = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", s)
        if m_img:
            close_lists()
            stem = Path(m_img.group(2)).stem
            # Look ahead for figure caption
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            cap = ""
            if j < len(lines) and re.match(r"^\*\*Figure \d+\.\*\*", lines[j].strip()):
                cap = re.sub(r"^\*\*Figure \d+\.\*\*\s*", "", lines[j].strip())
                i = j + 1
            else:
                i += 1
            out += [
                r"\begin{figure}[t]",
                r"\centering",
                f"\\includegraphics[width=0.96\\linewidth]{{figures/{stem}.pdf}}",
                f"\\caption{{{tex_escape_inline(cap)}}}",
                f"\\label{{fig:{stem}}}",
                r"\end{figure}",
                "",
            ]
            continue

        m_tcap = re.match(r"^\*\*(Table ([A-Z]?\d+))\.\*\*\s*(.+)$", s)
        if m_tcap:
            close_lists()
            pending_table_label = m_tcap.group(2).lower()
            pending_table_cap = m_tcap.group(3)
            i += 1
            continue

        if s.startswith("|"):
            close_lists()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                parts = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.match(r"^:?-+:?$", p) for p in parts):
                    rows.append(parts)
                i += 1
            ncols = len(rows[0])
            # Determine per-column alignment cleanly (left for text, right for numeric, p{} for long prose)
            col_specs = []
            for ci in range(ncols):
                col_vals = [strip_md_inline(r[ci]) for r in rows[1:] if ci < len(r)]
                max_len = max((len(v) for v in col_vals), default=0)
                num_hits = sum(1 for v in col_vals if re.match(r"^[+\-−]?\d|^\.\d|^\[|^—$", v))
                if ci == 0:
                    col_specs.append(r"p{0.26\linewidth}" if max_len > 36 else "l")
                elif max_len > 55:
                    col_specs.append(r"p{0.46\linewidth}")
                elif col_vals and num_hits >= 0.6 * len(col_vals):
                    col_specs.append("r")
                else:
                    col_specs.append("l")
            colspec = "".join(col_specs)
            use_resize = not any("p{" in cs for cs in col_specs)
            out += [
                r"\begin{table}[t]",
                r"\centering",
                r"\footnotesize" if ncols >= 7 else r"\small",
                f"\\caption{{{tex_escape_inline(pending_table_cap or '')}}}",
                f"\\label{{tab:{pending_table_label or 't'}}}",
            ]
            if use_resize:
                out.append(r"\resizebox{\linewidth}{!}{%")
            out += [
                f"\\begin{{tabular}}{{{colspec}}}",
                r"\toprule",
                " & ".join(f"\\textbf{{{tex_escape_inline(c)}}}" for c in rows[0]) + r" \\",
                r"\midrule",
            ]
            for r in rows[1:]:
                out.append(" & ".join(tex_escape_inline(c) for c in r) + r" \\")
            out += [
                r"\bottomrule",
                r"\end{tabular}" + ("%" if use_resize else ""),
            ]
            if use_resize:
                out.append(r"}")
            out += [
                r"\end{table}",
                "",
            ]
            pending_table_cap = None
            pending_table_label = None
            continue

        if s.startswith("- "):
            if in_enum:
                out.append(r"\end{enumerate}")
                in_enum = False
            if not in_itemize:
                out.append(r"\begin{itemize}")
                in_itemize = True
            out.append(r"\item " + tex_escape_inline(s[2:].strip()))
            i += 1
            continue

        m_num = re.match(r"^\d+\.\s+(.+)$", s)
        if m_num:
            if in_itemize:
                out.append(r"\end{itemize}")
                in_itemize = False
            if not in_enum:
                out.append(r"\begin{enumerate}")
                in_enum = True
            out.append(r"\item " + tex_escape_inline(m_num.group(1)))
            i += 1
            continue

        close_lists()
        out.append(tex_escape_inline(s) + "\n")
        i += 1

    close_lists()
    if not is_supp:
        out += [
            r"\bibliographystyle{plainnat}",
            r"\bibliography{UARCap_References}",
        ]
    out.append(r"\end{document}")
    tex_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Built LaTeX: {tex_path.name}")


def build_latex_readme():
    txt = """# LaTeX Compilation Guide (UARCap v0.8)

This folder contains self-contained, journal-neutral LaTeX source files for the main manuscript (`main.tex`), supplementary materials (`supplementary.tex`), verified BibTeX database (`UARCap_References.bib`, 31 references), and vector PDF figures (`figures/fig1_framework.pdf` through `figures/fig5_forest.pdf`).

## Standard Compilation (`pdflatex` + `bibtex`)

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

For Supplementary Materials:

```bash
pdflatex supplementary.tex
pdflatex supplementary.tex
```

## Notes for Journal Template Adaptation

- Author list, affiliations, corresponding email, and Declarations contain explicit `[TO BE COMPLETED BY AUTHORS]` placeholders.
- Replace `\\documentclass[11pt,a4paper]{article}` and `\\bibliographystyle{plainnat}` with the target journal class (e.g., `elsarticle`, `sn-jnl`, `acmart`, or `IEEEtran`) once the target venue is finalized.
"""
    (TEX_OUT / "README_LATEX.md").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    ensure_dirs()
    build_verified_bib()
    copy_figures_and_markdown()
    export_tables()
    md_to_docx(
        SUB / "UARCap_Manuscript_v0.8.md",
        SUB / "UARCap_Manuscript_v0.8.docx",
        "When Does External Retrieval Help Video Captioning?",
        FIG_OUT,
        is_supp=False,
    )
    md_to_docx(
        SUB / "UARCap_Supplementary_Materials.md",
        SUB / "UARCap_Supplementary_Materials.docx",
        "Supplementary Materials — External Retrieval Video Captioning",
        FIG_OUT,
        is_supp=True,
    )
    build_cover_letter()
    md_to_latex(SUB / "UARCap_Manuscript_v0.8.md", TEX_OUT / "main.tex", is_supp=False)
    md_to_latex(SUB / "UARCap_Supplementary_Materials.md", TEX_OUT / "supplementary.tex", is_supp=True)
    build_latex_readme()
    print("Submission v0.8 build complete.")
