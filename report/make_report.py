"""Build the BCSE497J Project-I report in the VIT template (report/CardioSentinel_Project_Report.docx + .pdf).

Fills the template's front matter in place (cover, declaration, certificate, acknowledgements, contents, lists,
abbreviations, symbols, abstract) and replaces its sample chapters with the content of report/content.py.
Results tables and figure captions are read from the paper's LaTeX sources, so they match the paper exactly.
Page numbers in the contents and lists are found by rendering the document with LibreOffice (two passes).
Placeholders whose values are not in the repository (guide, HoD, programme) stay highlighted.

Needs python-docx, Pillow, tectonic, pdftoppm/pdftotext and LibreOffice (soffice). Run make_onecolumn.py and
make_docx.py first (their build provides the architecture figure, workflow figure and algorithm images).

    python report/make_report.py --template "~/Downloads/BCSE497J Project I Report - Template FINAlL Updated  (2).docx"
    python report/make_report.py --debug        # one pass, no PDF and no page numbers
"""
from __future__ import annotations

import argparse
import copy
import re
import shutil
import subprocess
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
PAPER = ROOT / "paper"
BUILD = REPORT / "build"
sys.path[:0] = [str(REPORT), str(PAPER / "scripts")]

import content as C  # noqa: E402
import diagrams  # noqa: E402
from make_docx import arg_end, inline_inputs, read_aux, take_macro  # noqa: E402

DEFAULT_TEMPLATE = Path.home() / "Downloads" / "BCSE497J Project I Report - Template FINAlL Updated  (2).docx"
FONT = "Times New Roman"
TEXT_W = 6.45                      # usable width of the chapter section, inches
SUB, SUP, END = "\x01", "\x02", "\x03"   # markers for _{..} and ^{..} while converting LaTeX

# ---------------------------------------------------------------- LaTeX -> markup
MATH = {r"\pm": "±", r"\approx": "≈", r"\phi": "φ", r"\Delta": "Δ", r"\mu": "μ", r"\beta": "β", r"\tau": "τ",
        r"\sigma": "σ", r"\ell": "ℓ", r"\alpha": "α", r"\theta": "θ", r"\in": "∈", r"\le": "≤", r"\ge": "≥",
        r"\leq": "≤", r"\geq": "≥", r"\times": "×", r"\rightarrow": "→", r"\to": "→", r"\sum": "Σ", r"\cdot": "·",
        r"\dots": "…", r"\ldots": "…", r"\star": "*", r"\infty": "∞", r"\mid": "|", r"\langle": "⟨",
        r"\rangle": "⟩", r"\emptyset": "∅", r"\cup": "∪", r"\setminus": "∖", r"\subseteq": "⊆", r"\gets": "←"}


def math_to_markup(m: str) -> str:
    for cmd in ("mathrm", "text", "mathbf", "operatorname", "mathcal", "textit", "textbf"):
        m = re.sub(r"\\" + cmd + r"\{([^{}]*)\}", r"\1", m)
    m = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", m)
    for k in sorted(MATH, key=len, reverse=True):
        m = re.sub(re.escape(k) + r"(?![A-Za-z])", MATH[k], m)
    m = re.sub(r"\\[,;!:]|\\quad|\\qquad|\\left|\\right|\\big|\\Big", " ", m)
    m = re.sub(r"_\{([^{}]*)\}", SUB + r"\1" + END, m)
    m = re.sub(r"\^\{([^{}]*)\}", SUP + r"\1" + END, m)
    m = re.sub(r"_(\\?\w)", SUB + r"\1" + END, m)
    m = re.sub(r"\^(\\?\w|[−+*])", SUP + r"\1" + END, m)
    m = m.replace("{", "").replace("}", "").replace("-", "−")
    return re.sub(r"\s+", " ", m).strip()


def latex_to_markup(s: str, cites: dict) -> str:
    s = re.sub(r"(?<!\\)%.*", "", s)
    s = re.sub(r"\$([^$]*)\$", lambda m: math_to_markup(m.group(1)), s)
    s = re.sub(r"\\cite\{([^}]*)\}", lambda m: ", ".join(f"[{cites[k.strip()]}]" for k in m.group(1).split(",")), s)
    s = re.sub(r"\\ref\{([^}]*)\}", r"⟦\1⟧", s)          # resolved later through the report's labels
    for cmd, mark in (("textbf", "**"), ("bfseries", "**"), ("emph", "*"), ("textit", "*")):
        s = re.sub(r"\\" + cmd + r"\s*\{((?:[^{}]|\{[^{}]*\})*)\}", lambda m: mark + m.group(1) + mark, s)
    s = re.sub(r"\\textsuperscript\{([^}]*)\}", SUP + r"\1" + END, s)
    s = re.sub(r"\\(texttt|url|mbox)\{([^}]*)\}", r"\2", s)
    for a, b in ((r"\"a", "ä"), (r"\"o", "ö"), (r"\"u", "ü"), (r"\'e", "é"), (r"\'a", "á"), (r"\~a", "ã")):
        s = s.replace("{" + a + "}", b).replace(a, b)
    s = (s.replace("---", "—").replace("--", "–").replace("``", "“").replace("''", "”").replace(r"\%", "%")
         .replace(r"\&", "&").replace(r"\#", "#").replace(r"\_", "_").replace("~", " ").replace(r"\,", " ")
         .replace(r"\ ", " ").replace(r"\newblock", " ").replace(r"\ldots", "…"))
    s = re.sub(r"\\(BIBentry\w+|footnotesize|scriptsize|small|tiny|centering|relax|hskip[^\\]*)\b", "", s)
    s = re.sub(r"\\[A-Za-z]+\*?", "", s)
    s = s.replace("{", "").replace("}", "")
    s = s.replace(SUB, "_{").replace(SUP, "^{").replace(END, "}")
    return re.sub(r"\s+", " ", s).strip()


def split_top(s: str, sep: str) -> list[str]:
    """Split on sep outside braces (sep '\\\\' or '&'; escaped \\& ignored)."""
    out, depth, cur, i = [], 0, "", 0
    while i < len(s):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if depth == 0 and s.startswith(sep, i) and not (sep == "&" and i and s[i - 1] == "\\"):
            out.append(cur)
            cur, i = "", i + len(sep)
            continue
        cur += ch
        i += 1
    out.append(cur)
    return out


def parse_tabular(tex: str, cites: dict) -> tuple[list, list]:
    body = tex[tex.index(r"\begin{tabular}"):tex.index(r"\end{tabular}")].replace("@{}", "")
    body = body[arg_end(body, body.index("{", len(r"\begin{tabular}"))):]
    head_rows, rows, seen_mid = [], [], False
    for raw in split_top(body, "\\\\"):
        line = re.sub(r"\\(toprule|bottomrule)", "", raw)
        line = re.sub(r"\\cmidrule(\([^)]*\))?\{[^}]*\}", "", line)
        # after splitting on \\, a \midrule opens the segment of the row that follows it
        seen_mid = seen_mid or r"\midrule" in line
        line = line.replace(r"\midrule", "").strip()
        if not line:
            continue
        m = re.match(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}\s*$", line, re.S)
        if m and seen_mid and "&" not in line.replace(r"\&", ""):
            rows.append(("span", latex_to_markup(m.group(2), cites)))
            continue
        cells = []
        for c in split_top(line, "&"):
            mc = re.match(r"\s*\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}\s*$", c, re.S)
            n, txt = (int(mc.group(1)), mc.group(2)) if mc else (1, c)
            cells += [latex_to_markup(txt, cites)] * n
        (rows if seen_mid else head_rows).append(cells)
    # several header rows (column groups) -> one header row: "Group Column"
    header = [" ".join(dict.fromkeys(h[i] for h in head_rows if i < len(h) and h[i])).strip()
              for i in range(max(len(h) for h in head_rows))]
    return header, rows


def paper_float(label: str) -> tuple[str, str]:
    """(caption, float body with \\input expanded) of the paper float carrying this label."""
    src = inline_inputs("\n".join((PAPER / "sections" / f).read_text() for f in
                                  ("related_work.tex", "proposed_model.tex", "results.tex")), PAPER)
    i = src.index(r"\label{" + label + "}")
    start = max(src.rfind(r"\begin{table", 0, i), src.rfind(r"\begin{figure", 0, i))
    env = re.match(r"\\begin\{(\w+\*?)\}", src[start:]).group(1)
    end = src.index(r"\end{" + env + "}", i)
    body, cap = take_macro(src[start:end], "caption")
    return cap, body


# ---------------------------------------------------------------- docx helpers
def set_font(run, size=None, bold=None, italic=None, name=FONT):
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(a), name)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def add_runs(p, text: str, size=12, bold=False, italic=False, highlight=False, font=FONT):
    """Markup -> runs: **bold**, *italic*, _{sub}, ^{sup}."""
    tokens = re.split(r"(\*\*|\*|_\{[^}]*\}|\^\{[^}]*\})", text)
    b, it = bold, italic
    for t in tokens:
        if t == "**":
            b = not b
            continue
        if t == "*":
            it = not it
            continue
        if not t:
            continue
        sub = t.startswith("_{")
        sup = t.startswith("^{")
        r = p.add_run(t[2:-1] if (sub or sup) else t)
        set_font(r, size, b, it, font)
        r.font.subscript, r.font.superscript = sub or None, sup or None
        if highlight:
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
    return p


def fmt(p, align=WD_ALIGN_PARAGRAPH.JUSTIFY, spacing=1.15, before=0, after=6, first=None, left=None):
    pf = p.paragraph_format
    p.alignment = align
    pf.line_spacing, pf.space_before, pf.space_after = spacing, Pt(before), Pt(after)
    if first is not None:
        pf.first_line_indent = Inches(first)
    if left is not None:
        pf.left_indent = Inches(left)
    return p


def outline(p, level: int):
    ppr = p._p.get_or_add_pPr()
    o = OxmlElement("w:outlineLvl")
    o.set(qn("w:val"), str(level))
    ppr.append(o)


def clear(p):
    for r in list(p.runs):
        r._r.getparent().remove(r._r)
    for h in p._p.findall(qn("w:hyperlink")):
        p._p.remove(h)


def rewrite(p, segments, size=12):
    """Replace a template paragraph's text, keeping its paragraph format. segments: [(markup, highlight)]."""
    base = p.runs[0] if p.runs else None
    bold = base.bold if base is not None else False
    sz = base.font.size.pt if base is not None and base.font.size else size
    clear(p)
    for text, hl in segments:
        add_runs(p, text, size=sz, bold=bool(bold), highlight=hl)


def delete(p):
    p._p.getparent().remove(p._p)


def borders(tbl, header_rows=1):
    tblPr = tbl._tbl.tblPr
    for old in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(old)
    b = OxmlElement("w:tblBorders")
    for side in ("top", "bottom"):
        e = OxmlElement(f"w:{side}")
        e.set(qn("w:val"), "single"), e.set(qn("w:sz"), "10"), e.set(qn("w:color"), "000000")
        b.append(e)
    tblPr.append(b)
    for row in tbl.rows[:header_rows]:
        for cell in row.cells:
            tcPr = cell._tc.get_or_add_tcPr()
            cb = OxmlElement("w:tcBorders")
            e = OxmlElement("w:bottom")
            e.set(qn("w:val"), "single"), e.set(qn("w:sz"), "6"), e.set(qn("w:color"), "000000")
            cb.append(e)
            tcPr.append(cb)


# ---------------------------------------------------------------- assets
def asset(ref: str, shots: Path) -> Path:
    kind, name = ref.split(":", 1)
    if kind == "diagram":
        p = BUILD / "diagrams" / f"{name}.png"
        if not p.exists():
            diagrams.render(name, diagrams.DIAGRAMS[name], BUILD / "diagrams")
        return p
    if kind == "paperfig" and Path(name).suffix:      # image file used as-is (e.g. confusion_clients.jpg)
        return PAPER / "figures" / name
    if kind == "paperfig":
        p = BUILD / "figures" / f"{name}.png"
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["pdftoppm", "-png", "-r", "250", "-singlefile", str(PAPER / "figures" / f"{name}.pdf"),
                            str(p.with_suffix(""))], check=True)
        return p
    if kind == "paperpng":
        p = PAPER / "onecolumn" / "docx_build" / f"{name}.png"
        if not p.exists():
            raise SystemExit(f"{p} missing: run paper/scripts/make_onecolumn.py and make_docx.py first")
        return p
    if kind == "shot":
        return shots / f"{name}.png"
    raise ValueError(ref)


def render_eq(key: str, latex: str) -> Path:
    d = BUILD / "equations" / key.replace(":", "_")
    png = d.with_suffix(".png")
    if png.exists():
        return png
    d.mkdir(parents=True, exist_ok=True)
    (d / "e.tex").write_text("\\documentclass[12pt,border=1pt]{standalone}\n\\usepackage{amsmath}\n"
                             "\\usepackage{newtxtext,newtxmath}\n\\begin{document}\n$\\displaystyle " + latex +
                             "$\n\\end{document}\n")
    subprocess.run(["tectonic", "e.tex"], cwd=d, check=True, capture_output=True)
    subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", "e.pdf", str(png.with_suffix(""))], cwd=d,
                   check=True)
    return png


# ---------------------------------------------------------------- numbering
def number_blocks(blocks, cites):
    """Assign chapter-based numbers; return (numbers by key, toc entries, figures, tables)."""
    nums, toc, figs, tabs = {}, [], [], []
    ch = h2 = h3 = fi = ti = ei = 0
    for b in blocks:
        kind = b[0]
        if kind == "chapter":
            ch, h2, h3, fi, ti, ei = ch + 1, 0, 0, 0, 0, 0
            toc.append(("chapter", f"{ch}.", b[1].upper(), f"CHAPTER {ch}"))
        elif kind == "h2":
            h2, h3 = h2 + 1, 0
            toc.append(("section", "", f"{ch}.{h2} {b[1]}", f"{ch}.{h2} {b[1]}"))
            nums[f"sec:{b[1]}"] = f"{ch}.{h2}"
        elif kind == "h3":
            h3 += 1
            toc.append(("subsection", "", f"{ch}.{h2}.{h3} {b[1]}", f"{ch}.{h2}.{h3} {b[1]}"))
        elif kind == "fig":
            fi += 1
            nums[b[1]] = f"Figure {ch}.{fi}"
            figs.append((f"{ch}.{fi}", b[1]))
        elif kind == "table":
            ti += 1
            nums[b[1]] = f"Table {ch}.{ti}"
            tabs.append((f"{ch}.{ti}", b[1]))
        elif kind == "eq":
            ei += 1
            nums[b[1]] = f"({ch}.{ei})"
    nums["sec:cost"] = nums["sec:Cost Analysis"]
    return nums, toc, figs, tabs


PAPER_REFS = {"sec:discussion": "sec:Comparison with the State of the Art"}   # paper label -> report key


def resolve(text: str, nums: dict) -> str:
    text = re.sub(r"⟦([^⟧]+)⟧", lambda m: nums[PAPER_REFS[m.group(1)]], text)
    return re.sub(r"\{((?:fig|tab|eq|sec):[^}]+)\}", lambda m: nums[m.group(1)], text)


# ---------------------------------------------------------------- body writer
class Writer:
    def __init__(self, doc, anchor, nums, cites, shots):
        self.doc, self.anchor, self.nums, self.cites, self.shots = doc, anchor, nums, cites, shots
        self.captions = {}          # key -> caption markup (for the lists)
        self.first_in_section = True

    def para(self, text="", **kw):
        p = self.anchor.insert_paragraph_before("")
        if text:
            add_runs(p, resolve(text, self.nums), size=kw.pop("size", 12), bold=kw.pop("bold", False),
                     italic=kw.pop("italic", False))
        return fmt(p, **kw)

    def chapter(self, n, title, page_break=True):
        p = self.para(f"CHAPTER {n}", size=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=6)
        p.paragraph_format.page_break_before = page_break
        outline(p, 0)
        t = self.para(title.upper(), size=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=18)
        t.paragraph_format.keep_with_next = True

    def heading(self, text, level):
        p = self.para(text, size=14 if level == 1 else 12, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT,
                      before=12 if level == 1 else 8, after=6)
        p.paragraph_format.keep_with_next = True
        outline(p, level)

    def items(self, texts, numbered):
        for i, t in enumerate(texts, 1):
            p = self.para("", left=0.4, first=-0.3, after=3)
            p.paragraph_format.tab_stops.add_tab_stop(Inches(0.4))
            add_runs(p, (f"{i}.\t" if numbered else "•\t") + resolve(t, self.nums))

    def caption(self, label, text):
        p = self.para("", align=WD_ALIGN_PARAGRAPH.CENTER, before=4, after=10, spacing=1.0)
        add_runs(p, f"{label} ", size=10, bold=True)
        add_runs(p, resolve(text, self.nums), size=10)
        return p

    def figure(self, key, ref, cap, width):
        if cap is None:
            cap = latex_to_markup(paper_float(key)[0], self.cites)
        img = asset(ref, self.shots)
        p = self.para("", align=WD_ALIGN_PARAGRAPH.CENTER, before=6, after=2, spacing=1.0)
        p.paragraph_format.keep_with_next = True
        from PIL import Image
        w, h = Image.open(img).size
        width = min(width, TEXT_W, 8.3 * w / h)          # keep tall images on one page
        p.add_run().add_picture(str(img), width=Inches(width))
        self.caption(self.nums[key], cap)
        self.captions[key] = cap

    def table(self, key, src, cap):
        if src[0] == "tex":
            tcap, body = paper_float(src[1])
            header, rows = parse_tabular(body, self.cites)
            cap = cap or latex_to_markup(tcap, self.cites)
        else:
            header, rows = src[1], src[2]
        cap = cap[0].upper() + cap[1:]
        c = self.caption(self.nums[key], cap)
        c.paragraph_format.keep_with_next = True
        self.captions[key] = cap
        ncol = len(header)
        tbl = self.doc.add_table(rows=1 + len(rows), cols=ncol)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        need = [max(8, min(len(header[i]), 30)) for i in range(ncol)]
        for r in rows:
            if r[0] != "span":
                for i, cell in enumerate(r[:ncol]):
                    need[i] = max(need[i], min(len(cell), 42))
        widths = [TEXT_W * n / sum(need) for n in need]
        floor = 0.6                                      # inches; short columns such as "[7]" or "This work"
        short_cols = [i for i, w_ in enumerate(widths) if w_ < floor]
        if short_cols:
            rest = TEXT_W - floor * len(short_cols)
            long_sum = sum(w_ for i, w_ in enumerate(widths) if i not in short_cols)
            widths = [floor if i in short_cols else w_ * rest / long_sum for i, w_ in enumerate(widths)]

        def fill(cell, text, bold=False, italic=False):
            p = cell.paragraphs[0]
            fmt(p, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.0, after=1)
            add_runs(p, resolve(text, self.nums), size=10, bold=bold, italic=italic)

        for i, h in enumerate(header):
            fill(tbl.rows[0].cells[i], h, bold=True)
        for ri, r in enumerate(rows, 1):
            cells = tbl.rows[ri].cells
            if r[0] == "span":
                m = cells[0].merge(cells[-1])
                fill(m, r[1].strip("*"), italic=True)
            else:
                for i in range(ncol):
                    fill(cells[i], r[i] if i < len(r) else "")
        for row in tbl.rows:
            for i, cell in enumerate(row.cells):
                cell.width = Inches(widths[min(i, ncol - 1)])
        for i, col in enumerate(tbl.columns):
            col.width = Inches(widths[i])
        borders(tbl)
        for i, row in enumerate(tbl.rows):               # rows never split; the header repeats on a new page
            trPr = row._tr.get_or_add_trPr()
            trPr.append(OxmlElement("w:cantSplit"))
            if i == 0:
                trPr.append(OxmlElement("w:tblHeader"))
        self.anchor._p.addprevious(tbl._tbl)
        self.para("", after=4, spacing=1.0)

    def equation(self, key, latex):
        png = render_eq(key, latex)
        from PIL import Image
        w, _ = Image.open(png).size
        p = self.para("", align=WD_ALIGN_PARAGRAPH.LEFT, before=4, after=8, spacing=1.0)
        ts = p.paragraph_format.tab_stops
        ts.add_tab_stop(Inches(TEXT_W / 2), WD_TAB_ALIGNMENT.CENTER)
        ts.add_tab_stop(Inches(TEXT_W), WD_TAB_ALIGNMENT.RIGHT)
        p.add_run("\t")
        p.add_run().add_picture(str(png), width=Inches(min(w / 300, TEXT_W - 0.8)))
        r = p.add_run("\t" + self.nums[key])
        set_font(r, 12)

    def code(self, text, size=9):
        for line in text.split("\n"):
            p = self.para("", align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.0, after=0, left=0.25)
            r = p.add_run(line if line else " ")
            set_font(r, size, name="Courier New")
            shd = OxmlElement("w:shd")
            shd.set(qn("w:val"), "clear"), shd.set(qn("w:fill"), "F2F2F2")
            p._p.get_or_add_pPr().append(shd)
        self.para("", after=4, spacing=1.0)

    def write(self, blocks):
        ch = h2 = 0
        for b in blocks:
            kind = b[0]
            if kind == "chapter":
                ch, h2 = ch + 1, 0
                self.chapter(ch, b[1])
            elif kind == "h2":
                h2 += 1
                h3 = 0
                self.heading(f"{ch}.{h2} {b[1]}", 1)
            elif kind == "h3":
                h3 += 1
                self.heading(f"{ch}.{h2}.{h3} {b[1]}", 2)
            elif kind == "p":
                self.para(b[1], first=0.5)
            elif kind in ("bullets", "numbered"):
                self.items(b[1], kind == "numbered")
            elif kind == "fig":
                self.figure(b[1], b[2], b[3], b[4])
            elif kind == "table":
                self.table(b[1], b[2], b[3] if len(b) > 3 else None)
            elif kind == "eq":
                self.equation(b[1], b[2])
            elif kind == "code":
                self.code(b[1])


# ---------------------------------------------------------------- front matter
def short(cap: str, n: int = 95) -> str:
    cap = re.sub(r"\*\*|\*", "", cap)
    cut = re.split(r" \(|; |: ", cap)[0]
    return cut if len(cut) <= n else cut[: n - 1].rsplit(" ", 1)[0] + "…"


def clone_after(el, template_el):
    new = copy.deepcopy(template_el)
    el.addnext(new)
    return new


def set_cell(cell, text, bold=None, size=12):
    p = cell.paragraphs[0]
    base = p.runs[0] if p.runs else None
    b = base.bold if base is not None and bold is None else bold
    clear(p)
    for extra in cell.paragraphs[1:]:
        delete(extra)
    add_runs(p, text, size=size, bold=bool(b))


def front_matter(doc, P, T, toc, figs, tabs, captions, pages):
    title = C.TITLE.upper()
    # cover
    rewrite(P[2], [(title, False)])
    delete(P[3])
    rewrite(P[12], [("Programme", True)])
    delete(P[16])
    stud = T[0]
    for row, (reg, name) in zip(stud.rows, C.STUDENTS):
        set_cell(row.cells[0], reg, bold=True)
        set_cell(row.cells[1], name.upper(), bold=True)
    stud._tbl.remove(stud.rows[2]._tr)
    for row in T[1].rows[:2]:
        p = row.cells[0].paragraphs[0]
        for r in p.runs:
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
    names = " and ".join(f"{n} ({r})" for r, n in C.STUDENTS)
    # declaration
    rewrite(P[27], [(f"We hereby declare that the project entitled “{C.TITLE}” submitted by us, for the award of the "
                     "degree of *Bachelor of Technology in Computer Science and Engineering* to VIT is a record of "
                     "bonafide work carried out by us under the supervision of Prof. / Dr. ", False),
                    ("<< Guide Name >>", True), (".", False)])
    rewrite(P[28], [("We further declare that the work reported in this project has not been submitted and will not "
                     "be submitted, either in part or in full, for the award of any other degree or diploma in this "
                     "institute or any other institute or university.", False)])
    rewrite(P[37], [("Signature of the Candidates", False)])
    P[37].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    delete(P[43])
    # certificate
    rewrite(P[47], [(f"This is to certify that the project entitled “{C.TITLE}” submitted by **{names}**, **School of "
                     "Computer Science and Engineering**, VIT, for the award of the degree of *Bachelor of Technology "
                     "in Computer Science and Engineering*, is a record of bonafide work carried out by them under my "
                     "supervision during Fall Semester 2026-2027, as per the VIT code of academic and research "
                     "ethics.", False)])
    rewrite(P[49], [("The contents of this report have not been submitted and will not be submitted either in part or "
                     "in full, for the award of any other degree or diploma in this institute or any other institute "
                     "or university. The project fulfills the requirements and regulations of the University and in "
                     "my opinion meets the necessary standards for submission.", False)])
    delete(P[69])
    for i in (51, 52, 60, 61, 62):           # longer text: drop blank lines so HoD / Programme stay on the page
        if not P[i].text.strip():
            delete(P[i])
    for i in (73, 74):
        for r in P[i].runs:
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
    # acknowledgements
    rewrite(P[79], [("We are deeply grateful to the management of Vellore Institute of Technology (VIT) for providing "
                     "us with the opportunity and resources to undertake this project. Their commitment to fostering "
                     "a conducive learning environment has been instrumental in our academic journey. The support and "
                     "infrastructure provided by VIT have enabled us to explore and develop our ideas to their fullest "
                     "potential.", False)])
    rewrite(P[81], [("Our sincere thanks to Dr. Jaisankar N, the Dean of the School of Computer Science and "
                     "Engineering (SCOPE), for his unwavering support and encouragement. His leadership and vision have "
                     "greatly inspired us to strive for excellence. The Dean’s dedication to academic excellence and "
                     "innovation has been a constant source of motivation for us. We appreciate his efforts in creating "
                     "an environment that nurtures creativity and critical thinking.", False)])
    rewrite(P[83], [("We express our profound appreciation to ", False), ("<Head of Department’s Name>", True),
                    (", the Head of the ", False), ("<Department Name>", True),
                    (", for his/her insightful guidance and continuous support. His/her expertise and advice have been "
                     "crucial in shaping the direction of our project. The Head of Department’s commitment to "
                     "fostering a collaborative and supportive atmosphere has greatly enhanced our learning "
                     "experience. His/her constructive feedback and encouragement have been invaluable in overcoming "
                     "challenges and achieving our project goals.", False)])
    rewrite(P[85], [("We are immensely thankful to our project guide, ", False), ("<Guide’s Name>", True),
                    (", for his/her dedicated mentorship and invaluable feedback. His/her patience, knowledge, and "
                     "encouragement have been pivotal in the successful completion of this project. Our supervisor’s "
                     "willingness to share his/her expertise and provide thoughtful guidance has been instrumental in "
                     "refining our ideas and methodologies. His/her support has not only contributed to the success of "
                     "this project but has also enriched our overall academic experience.", False)])
    rewrite(P[91], [(C.STUDENTS[0][1], False)])
    second = clone_after(P[91]._p, P[91]._p)
    from docx.text.paragraph import Paragraph
    rewrite(Paragraph(second, P[91]._parent), [(C.STUDENTS[1][1], False)])

    # table of contents
    delete(P[94])
    toc_t = T[2]
    rows = toc_t.rows
    chap_tr, sec_tr = rows[2]._tr, rows[3]._tr
    for r in rows[2:]:
        toc_t._tbl.remove(r._tr)
    set_cell(rows[1].cells[1], "Abstract", bold=True)
    set_cell(rows[1].cells[2], pages.get("ABSTRACT", "00"), bold=True)
    from docx.table import _Row
    entries = list(toc) + [("chapter", "9.", "REFERENCES", "CHAPTER 9"), ("chapter", "", "APPENDIX A – SAMPLE CODE",
                                                                         "APPENDIX A")]
    for kind, sl, text, key in entries:
        tr = copy.deepcopy(chap_tr if kind == "chapter" else sec_tr)
        toc_t._tbl.append(tr)
        row = _Row(tr, toc_t)
        set_cell(row.cells[0], sl, bold=True)
        indent = "      " if kind == "subsection" else ""
        set_cell(row.cells[1], indent + text, bold=kind == "chapter")
        set_cell(row.cells[2], pages.get(key, "00"), bold=True)

    # the contents now run over two pages: start the list of figures with its own page break instead of the
    # template's break after the table (which would leave a blank page)
    for i in range(95, 99):                      # blank lines and the page break after the contents table
        if not P[i].text.strip():
            delete(P[i])
    P[99].paragraph_format.page_break_before = True

    # list of figures (table) and list of tables (paragraphs)
    lof = T[3]
    fig_tr = lof.rows[1]._tr
    for r in lof.rows[1:]:
        lof._tbl.remove(r._tr)
    for num, key in figs:
        tr = copy.deepcopy(fig_tr)
        lof._tbl.append(tr)
        row = _Row(tr, lof)
        set_cell(row.cells[0], num, bold=False)
        set_cell(row.cells[1], short(captions[key]), bold=False)
        set_cell(row.cells[2], pages.get(key, "00"), bold=False)
    delete(P[105])
    last = P[110]._p
    for num, key in tabs:
        new = clone_after(last, P[110]._p)
        p = Paragraph(new, P[110]._parent)
        base_sz = 12
        clear(p)
        add_runs(p, f"{num}\t{short(captions[key])}\t{pages.get(key, '00')}", size=base_sz)
        last = new
    delete(P[110])

    # abbreviations and symbols: hanging indent at the template's tab, so long meanings wrap under the meaning
    def hang(p):
        p.paragraph_format.tab_stops.add_tab_stop(Inches(2.6))
        p.paragraph_format.left_indent, p.paragraph_format.first_line_indent = Inches(2.6), Inches(-2.6)

    last = P[114]._p
    for ab, full in sorted(C.ABBREVIATIONS, key=lambda x: x[0].lower()):
        new = clone_after(last, P[114]._p)
        p = Paragraph(new, P[114]._parent)
        clear(p)
        add_runs(p, f"{ab}\t{full}")
        hang(p)
        last = new
    for i in list(range(114, 122)) + [128]:
        delete(P[i])
    last = P[132]._p
    for sym, meaning in C.SYMBOLS:
        new = clone_after(last, P[132]._p)
        p = Paragraph(new, P[132]._parent)
        clear(p)
        add_runs(p, f"{sym}\t{meaning}")
        hang(p)
        last = new
    delete(P[132]), delete(P[133])
    for i in range(134, 184):
        if not P[i]._p.xpath(".//w:sectPr|.//w:br|.//w:drawing"):
            delete(P[i])

    # abstract
    P[184].paragraph_format.page_break_before = True
    for i in range(185, 225):
        if not P[i]._p.xpath(".//w:sectPr|.//w:drawing"):
            delete(P[i])
    for text in C.ABSTRACT:
        p = P[225].insert_paragraph_before("")
        add_runs(p, text)
        fmt(p, spacing=1.15, after=6, first=0.5)


def find_pages(pdf: Path, toc, figs, tabs, captions) -> dict:
    n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(pdf)], capture_output=True,
                                                        text=True).stdout).group(1))
    pages = []
    for i in range(1, n + 1):
        t = subprocess.run(["pdftotext", "-f", str(i), "-l", str(i), str(pdf), "-"], capture_output=True,
                           text=True).stdout
        pages.append(re.sub(r"\s+", " ", t))
    decl = next(i for i, t in enumerate(pages) if "DECLARATION" in t)
    ch1 = next(i for i, t in enumerate(pages) if re.search(r"CHAPTER 1\b", t))
    roman = lambda k: ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv", "xv",
                       "xvi", "xvii", "xviii", "xix", "xx"][k]
    out = {"ABSTRACT": roman(next(i for i in range(decl, ch1) if "ABSTRACT" in pages[i]) - decl)}
    def words(s: str, n: int) -> str:                 # first n words, whitespace-tolerant pattern
        s = re.sub(r"\*\*|\*|_\{|\^\{|\}", "", s)
        return r"\s*".join(re.escape(w) for w in s.split()[:n])

    # headings in document order, searching forward page by page
    cur = ch1
    for _, _, _, key in toc:
        for i in range(cur, len(pages)):
            if re.search(words(key, 6), pages[i]):
                out[key] = str(i - ch1 + 1)
                cur = i
                break
    # captions: "Figure 2.1 <first words of the caption>" (not the first mention in the text)
    for items, word in ((figs, "Figure"), (tabs, "Table")):
        cur = ch1
        for num, key in items:
            pat = re.escape(f"{word} {num}") + r"\s+" + words(captions[key], 4)
            for i in range(cur, len(pages)):
                if re.search(pat, pages[i]):
                    out[key] = str(i - ch1 + 1)
                    cur = i
                    break
    for key, pat in (("CHAPTER 9", r"CHAPTER 9\b"), ("APPENDIX A", r"APPENDIX A")):
        out[key] = str(next(i for i in range(ch1, len(pages)) if re.search(pat, pages[i])) - ch1 + 1)
    return out


def build(template: Path, shots: Path, pages: dict, out: Path) -> None:
    labels, cites = read_aux(PAPER / "onecolumn" / "main.aux")
    blocks = C.chapters()
    nums, toc, figs, tabs = number_blocks(blocks, cites)
    doc = Document(template)
    P, T = list(doc.paragraphs), list(doc.tables)

    # chapters 1-8 go before the section break that ends the chapter section (template paragraph 230)
    for i in list(range(226, 230)) + list(range(231, 254)):
        delete(P[i])
    w = Writer(doc, P[230], nums, cites, shots)
    w.write(blocks)
    P[230].paragraph_format.page_break_before = False

    # chapter 9 (references) opens the last section, before the appendix page break (template paragraph 254)
    w.anchor = P[254]
    w.chapter(9, "References", page_break=False)
    bbl = (PAPER / "onecolumn" / "main.bbl").read_text()
    for i, item in enumerate(re.split(r"\\bibitem\{[^}]*\}", bbl)[1:], 1):
        item = item.replace(r"\end{thebibliography}", "")
        p = w.para("", left=0.45, first=-0.45, after=4, spacing=1.15)
        p.paragraph_format.tab_stops.add_tab_stop(Inches(0.45))
        add_runs(p, f"[{i}]\t" + latex_to_markup(item, cites))

    # appendix A after its heading (last paragraph of the template)
    w.anchor = doc.add_paragraph("")
    for title, path, a, b in C.APPENDIX_FILES:
        w.heading(title, 2)
        lines = (ROOT / path).read_text().splitlines()[a - 1:b]
        w.code("\n".join(lines).rstrip())
    delete(w.anchor)

    front_matter(doc, P, T, toc, figs, tabs, w.captions, pages)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    return toc, figs, tabs, w.captions


def to_pdf(docx: Path) -> Path:
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(docx.parent), str(docx)],
                   check=True, capture_output=True)
    return docx.with_suffix(".pdf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    ap.add_argument("--shots", type=Path, default=REPORT / "assets" / "screenshots")
    ap.add_argument("--debug", action="store_true", help="one pass, no PDF and no page numbers")
    args = ap.parse_args()
    BUILD.mkdir(parents=True, exist_ok=True)
    out = REPORT / "CardioSentinel_Project_Report.docx"
    if args.debug:
        build(args.template, args.shots, {}, BUILD / "debug.docx")
        print(f"wrote {BUILD / 'debug.docx'}")
        return
    first = BUILD / "pass1.docx"
    toc, figs, tabs, caps = build(args.template, args.shots, {}, first)
    pages = find_pages(to_pdf(first), toc, figs, tabs, caps)
    missing = [k for _, _, _, k in toc if k not in pages] + [k for _, k in figs + tabs if k not in pages]
    if missing:
        print("page not found for:", missing)
    build(args.template, args.shots, pages, out)
    pdf = to_pdf(out)
    check = find_pages(pdf, toc, figs, tabs, caps)
    moved = {k: (pages[k], check.get(k)) for k in pages if check.get(k) != pages[k]}
    print(f"wrote {out}\nwrote {pdf}\npage numbers stable: {not moved} {moved if moved else ''}")


if __name__ == "__main__":
    main()
