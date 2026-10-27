"""Build a single-column version of the paper from the unchanged two-column sources.

Copies main.tex, sections/, tables/ and references.bib into paper/onecolumn/, switches IEEEtran to
`onecolumn` (11 pt), and shrinks floats that were sized to one column (`\\columnwidth`) so they do not
stretch across the full page. Then compiles with Tectonic to paper/CardioSentinel_onecolumn.pdf.

    python paper/scripts/make_onecolumn.py
    python paper/scripts/make_onecolumn.py --debug      # write the sources only, no compile
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / "onecolumn"

FIG_SCALE = 0.68             # \includegraphics[width=s\columnwidth] -> s*0.68\textwidth
TAB_W = r"0.78\textwidth"   # \resizebox{\columnwidth}{!}{...} (tables, workflow diagram)


def convert(tex: str) -> str:
    def fig(m: re.Match) -> str:
        scale = float(m.group(1) or 1.0)
        return f"width={scale * FIG_SCALE:.2f}\\textwidth"
    tex = re.sub(r"width=([0-9.]*)\\columnwidth", fig, tex)
    return tex.replace(r"\resizebox{\columnwidth}", r"\resizebox{" + TAB_W + "}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true", help="write the sources only, do not compile")
    args = ap.parse_args()

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "sections").mkdir(parents=True)
    shutil.copytree(PAPER / "tables", OUT / "tables")
    shutil.copy(PAPER / "references.bib", OUT)
    for f in (PAPER / "sections").glob("*.tex"):
        (OUT / "sections" / f.name).write_text(convert(f.read_text()))

    main_tex = (PAPER / "main.tex").read_text()
    main_tex = main_tex.replace(r"\documentclass[journal]{IEEEtran}",
                                r"\documentclass[journal,onecolumn,11pt]{IEEEtran}")
    main_tex = main_tex.replace(r"\graphicspath{{figures/}}", r"\graphicspath{{../figures/}}")
    (OUT / "main.tex").write_text(main_tex)
    if args.debug:
        print(f"sources written to {OUT}")
        return

    # main.aux / main.bbl are kept for make_docx.py (label numbers and the reference list)
    subprocess.run(["tectonic", "--keep-logs", "--keep-intermediates", "main.tex"], cwd=OUT, check=True)
    shutil.copy(OUT / "main.pdf", PAPER / "CardioSentinel_onecolumn.pdf")
    print(f"wrote {PAPER / 'CardioSentinel_onecolumn.pdf'}")


if __name__ == "__main__":
    main()
