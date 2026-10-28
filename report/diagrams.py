"""Design diagrams for the project report (Chapter 4), drawn in TikZ and rendered to PNG.

Every box names a real module, class, route or file of the repository.

    python report/diagrams.py OUT_DIR
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PREAMBLE = r"""\documentclass[border=6pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{newtxtext}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,fit,calc,shapes.geometric,shapes.multipart,backgrounds}
\tikzset{>={Stealth[length=2.2mm]}, every node/.style={font=\small}}
\begin{document}
"""

DFD0 = r"""
\begin{tikzpicture}[
  ent/.style={draw, thick, minimum width=2.9cm, minimum height=1.1cm, align=center, fill=gray!8},
  proc/.style={draw, thick, circle, minimum size=3.4cm, align=center, fill=blue!6},
  lab/.style={font=\footnotesize, align=center, fill=white, inner sep=1.5pt}]
\node[proc] (p) at (0,0) {\textbf{0}\\[2pt]CardioSentinel\\ECG analysis\\system};
\node[ent] (phys) at (-7.0,1.3) {PhysioNet\\(PTB-XL v1.0.3)};
\node[ent] (kag) at (-7.0,-1.3) {Kaggle\\(ECG-Image-Database)};
\node[ent] (pat) at (0,4.6) {Patient};
\node[ent] (doc) at (0,-4.6) {Clinician};
\draw[->, thick] (phys.east) -- node[lab, above=2pt]{12-lead signals, SCP labels} ($(p.west)+(0,0.5)$);
\draw[->, thick] (kag.east) -- node[lab, below=2pt]{ECG photos and scans} ($(p.west)+(0,-0.5)$);
\draw[->, thick] ([xshift=-12pt]pat.south) -- node[lab, left=2pt]{ECG file / photo, symptoms} ([xshift=-12pt]p.north);
\draw[->, thick] ([xshift=12pt]p.north) -- node[lab, right=2pt]{verdict, findings, replies} ([xshift=12pt]pat.south);
\draw[->, thick] ([xshift=-12pt]p.south) -- node[lab, left=2pt]{AI pre-read, shared items} ([xshift=-12pt]doc.north);
\draw[->, thick] ([xshift=12pt]doc.north) -- node[lab, right=2pt]{sign-off, release, reply} ([xshift=12pt]p.south);
\end{tikzpicture}
"""

DFD1 = r"""
\begin{tikzpicture}[
  ent/.style={draw, thick, minimum width=2.5cm, minimum height=0.9cm, align=center, fill=gray!8},
  proc/.style={draw, thick, rounded corners=8pt, minimum width=3.3cm, minimum height=1.3cm, align=center, fill=blue!6},
  store/.style={draw=none, minimum width=3.3cm, minimum height=0.75cm, align=center, fill=yellow!10,
                path picture={\draw[thick] (path picture bounding box.north west) -- (path picture bounding box.north east)
                              (path picture bounding box.south west) -- (path picture bounding box.south east);}},
  lab/.style={font=\footnotesize, align=center, fill=white, inner sep=1pt}]
\node[ent] (phys) at (0,0) {PhysioNet\\PTB-XL};
\node[proc] (p1) at (4.2,0) {\textbf{1.0} Prepare data\\\scriptsize labels, band-pass};
\node[store] (d1) at (4.2,2.1) {D1 \;ptbxl\_signals.npy};
\node[proc] (p2) at (8.6,0) {\textbf{2.0} Train clients\\\scriptsize A, B, C, D (folds 1--8)};
\node[store] (d2) at (8.6,-2.1) {D2 \;outputs/fourclient};
\node[proc] (p3) at (13.0,0) {\textbf{3.0} Calibrate and\\aggregate (fold 9)};
\node[proc] (p4) at (13.0,-4.2) {\textbf{4.0} Evaluate\\(fold 10, once)};
\node[ent] (rep) at (17.0,-4.2) {Tables and\\figures};
\node[proc] (p5) at (8.6,-4.2) {\textbf{5.0} Portal inference\\\scriptsize digitize, clients, gate, triage};
\node[ent] (pat) at (4.2,-4.2) {Patient /\\Clinician};
\node[store] (d3) at (8.6,-6.3) {D3 \;cardiosentinel.db};
\node[ent] (kag) at (4.2,-6.3) {Kaggle ECG-\\Image-Database};
\draw[->, thick] (phys) -- node[lab, above]{raw records} (p1);
\draw[->, thick] (p1) -- (d1);
\draw[->, thick] (d1) -| node[lab, pos=0.25, above]{signals, labels} (p2.north);
\draw[->, thick] (p2) -- node[lab, right]{model.pt} (d2);
\draw[->, thick] (p2) -- node[lab, above=3pt]{$P_i$} (p3);
\draw[->, thick] (p3) -- node[lab, right]{weights, thresholds, bands} ($(p3.south)+(0,-1.0)$) |- ($(d2.east)+(0,0.15)$);
\draw[->, thick] (d2.east) ++(0,-0.15) -| node[lab, pos=0.75, right]{fold-10 predictions} (p4);
\draw[->, thick] (p4) -- (rep);
\draw[->, thick] (d2) -- node[lab, right]{trained models} (p5);
\draw[->, thick] ([yshift=5pt]pat.east) -- node[lab, above]{ECG / photo} ([yshift=5pt]p5.west);
\draw[->, thick] ([yshift=-5pt]p5.west) -- node[lab, below]{result} ([yshift=-5pt]pat.east);
\draw[<->, thick] (p5) -- node[lab, right]{patients, ECGs, reviews} (d3);
\draw[->, thick] (kag.north east) -- node[lab, sloped, above]{real photos} (p5.south west);
\end{tikzpicture}
"""

USECASE = r"""
\begin{tikzpicture}[
  uc/.style={draw, thick, ellipse, minimum width=4.2cm, minimum height=0.95cm, align=center, fill=blue!5, font=\footnotesize},
  lab/.style={font=\scriptsize\itshape, fill=white, inner sep=1pt}]
\newcommand{\actor}[3]{%
  \begin{scope}[shift={(#1)}]
    \draw[thick] (0,0.55) circle (0.25);
    \draw[thick] (0,0.3) -- (0,-0.45) (-0.4,0.1) -- (0.4,0.1) (0,-0.45) -- (-0.3,-0.95) (0,-0.45) -- (0.3,-0.95);
    \node[font=\small\bfseries] at (0,-1.25) {#2};
    \coordinate (#3) at (0,0);
  \end{scope}}
\node[uc] (u1) at (0,6.0) {Sign up / sign in};
\node[uc] (u2) at (0,4.8) {Check my heart\\(upload ECG, photo or PDF)};
\node[uc] (u3) at (0,3.6) {View result and findings};
\node[uc] (u4) at (0,2.4) {Log symptoms (journal)};
\node[uc] (u5) at (0,1.2) {Share with my doctor};
\node[uc] (u6) at (0,0.0) {Ask the voice assistant};
\node[uc] (u7) at (0,-1.2) {Book appointment / message};
\node[uc] (c1) at (6.4,4.8) {Review ECG queue};
\node[uc] (c2) at (6.4,3.6) {View 12-lead ECG};
\node[uc] (c3) at (6.4,2.4) {Sign off and release result};
\node[uc] (c4) at (6.4,1.2) {Reply to patient shares};
\node[uc] (c5) at (6.4,0.0) {Add an ECG for a patient};
\node[uc, fill=orange!10] (ai) at (3.2,-2.7) {Run AI pre-read\\(4 clients, gate, triage)};
\begin{scope}[on background layer]
  \node[draw, thick, rounded corners, fit=(u1)(u7)(c1)(c5)(ai), inner sep=10pt, label={[font=\small\bfseries]above:CardioSentinel portal}] {};
\end{scope}
\actor{-4.2,2.4}{Patient}{pa}
\actor{10.6,2.4}{Clinician}{cl}
\foreach \u in {u1,u2,u3,u4,u5,u6,u7} \draw (pa) -- (\u.west);
\foreach \u in {c1,c2,c3,c4,c5} \draw (cl) -- (\u.east);
\draw (cl) -- (u1.east);
\draw[->, dashed] (u2.south east) -- node[lab, sloped, above]{$\langle\langle$include$\rangle\rangle$} (ai.north west);
\draw[->, dashed] (c5.south west) -- node[lab, sloped, above]{$\langle\langle$include$\rangle\rangle$} (ai.north east);
\end{tikzpicture}
"""

CLASSES = r"""
\begin{tikzpicture}[
  cls/.style={draw, thick, rectangle split, rectangle split parts=3, fill=white, align=left, font=\scriptsize,
              rectangle split part fill={blue!8,white,white}, minimum width=3.6cm, text width=3.5cm},
  inh/.style={-{Triangle[open, length=2.6mm, width=2.4mm]}, thick},
  ass/.style={-{Stealth}, thick},
  comp/.style={{Diamond[length=2.6mm, width=2mm]}-, thick},
  lab/.style={font=\scriptsize, fill=white, inner sep=1pt}]
\node[cls] (nn) at (0,0) {\textbf{nn.Module}\nodepart{second}(PyTorch)\nodepart{third}forward()};
\node[cls] (base) at (0,-3.0) {\textbf{BaseClient}\nodepart{second}temperature: buffer\nodepart{third}encode(batch)\\forward(batch): ClientOutput};
\node[cls] (sc) at (0,-6.4) {\textbf{SignalClient}\nodepart{second}arch, backbone\\proj: Linear 256, head: Linear 5\nodepart{third}encode(batch)};
\node[cls] (cnn) at (-6.4,-10.2) {\textbf{CNN1D}\nodepart{second}4 conv blocks, k=7\nodepart{third}forward(x)};
\node[cls] (res) at (-2.15,-10.2) {\textbf{ResNet1D}\nodepart{second}stages [3,4,6,3]\nodepart{third}forward(x)};
\node[cls] (lstm) at (2.15,-10.2) {\textbf{BiLSTM}\nodepart{second}2 layers, 128 units\nodepart{third}forward(x)};
\node[cls] (cl) at (6.4,-10.2) {\textbf{CNNLSTM}\nodepart{second}3 conv + BiLSTM\nodepart{third}forward(x)};
\node[cls] (rb) at (-2.15,-13.2) {\textbf{ResBlock1d}\nodepart{second}conv, BN, skip\nodepart{third}forward(x)};
\node[cls] (eq) at (10.4,-0.4) {\textbf{EqualWeights}\nodepart{second}w = 1/4\nodepart{third}fit(), weights(), predict()};
\node[cls] (st) at (10.4,-3.3) {\textbf{StaticWeights}\nodepart{second}lr = 0.05, steps = 500\nodepart{third}fit(), weights(), predict()};
\node[cls] (dg) at (10.4,-6.4) {\textbf{DynamicGate}\nodepart{second}hidden 32, patience 30\nodepart{third}fit(), weights(), predict()};
\node[cls] (gn) at (10.4,-9.6) {\textbf{GateNet}\nodepart{second}MLP 24--32--4\nodepart{third}forward(P)};
\node[cls] (tr) at (5.2,-3.0) {\textbf{Triage}\nodepart{second}bands: lo, hi, mode per class\nodepart{third}decide(p)\\from\_dict(d)};
\node[cls] (eng) at (5.2,-6.4) {\textbf{Engine} (web)\nodepart{second}stacks, image\_stack\nodepart{third}predict(signal)\\predict\_images(files)};
\begin{scope}[on background layer]
  \node[draw, dashed, rounded corners, fit=(eq)(dg), inner sep=6pt, label={[font=\scriptsize]above:combiners (gate.py)}] (comb) {};
\end{scope}
\node[cls] (ds) at (-5.6,-3.0) {\textbf{SignalDataset}\nodepart{second}keys, augment\nodepart{third}\_\_getitem\_\_(i)};
\draw[inh] (base) -- (nn);
\draw[inh] (sc) -- (base);
\foreach \b in {cnn,res,lstm,cl} \draw[ass] (sc.south) -- (\b.north);
\node[lab] at (0,-8.3) {backbone (one of)};
\draw[comp] (res) -- node[lab, right]{1..16} (rb);
\draw[comp] (dg) -- (gn);
\draw[ass] (eng.west) -- node[lab, above]{uses 4} (sc.east);
\draw[ass] (eng) -- (tr);
\draw[ass] (eng.east) -- node[lab, above]{selected} (dg.west);
\draw[ass] (ds) -- node[lab, above]{batches} (base);
\end{tikzpicture}
"""

SEQUENCE = r"""
\begin{tikzpicture}[
  obj/.style={draw, thick, fill=blue!6, minimum width=2.4cm, minimum height=0.8cm, align=center, font=\footnotesize},
  msg/.style={->, thick}, ret/.style={->, dashed},
  lab/.style={font=\scriptsize, above, inner sep=1.5pt, align=center}]
\def\L{-12.6}
\foreach \x/\n/\t in {0/u/Patient, 3/b/Browser (SPA), 6/s/FastAPI server, 9/d/digitize\\(images), 12/e/Engine, 15/g/Clients A--D\\+ gate + Triage, 18/db/SQLite DB} {
  \node[obj] (\n) at (\x,0) {\t};
  \draw[dashed, gray] (\x,-0.45) -- (\x,\L);
}
\newcommand{\call}[4]{\draw[msg] (#1,#3) -- node[lab]{#4} (#2,#3);}
\newcommand{\back}[4]{\draw[ret] (#1,#3) -- node[lab]{#4} (#2,#3);}
\call{0}{3}{-1.2}{choose ECG file or photo}
\call{3}{6}{-2.0}{POST /api/me/analyze}
\draw[thick, fill=white] (5.9,-2.1) rectangle (6.1,-11.0);
\node[font=\scriptsize, align=left, anchor=west, fill=yellow!10, draw] at (6.2,-2.75) {alt: photo, scan or PDF};
\call{6}{9}{-3.5}{digitize(image)}
\back{9}{6}{-4.3}{(12, 1000) signal, quality}
\node[font=\scriptsize, align=left, anchor=west, fill=yellow!10, draw] at (6.2,-5.0) {else: CSV / NPY / WFDB};
\draw[thick] (6.1,-5.6) -- (6.8,-5.6) -- (6.8,-5.95);
\draw[msg] (6.8,-5.95) -- (6.1,-5.95);
\node[font=\scriptsize, anchor=west] at (6.9,-5.78) {parse\_upload(files)};
\call{6}{12}{-6.6}{predict(signal)}
\call{12}{15}{-7.4}{4 calibrated probabilities $P_i$}
\back{15}{12}{-8.2}{$P = \sum_i w_i P_i$, triage decisions}
\back{12}{6}{-9.0}{findings, verdict, weights}
\call{6}{18}{-9.8}{save ECG and result (private)}
\back{6}{3}{-10.9}{JSON result}
\back{3}{0}{-11.8}{verdict, findings, rhythm strip}
\end{tikzpicture}
"""

PLAN = r"""
\begin{tikzpicture}[x=0.105cm, y=0.62cm, bar/.style={fill=blue!45, draw=blue!70!black}]
\def\tasks{{"Literature review","Data download and pre-processing","Earlier 3-modality prototype","Redesign: four-client system","Training A-D, 3 seeds; aggregation","Baselines and per-finding triage","Portal, photo reading, XAI","Paper and report"}}
\foreach \x/\m in {0/July, 31/August, 62/September, 92/October} {
  \node[font=\scriptsize, anchor=west] at (\x+1, 0.7) {\m};
  \draw[gray!40] (\x,0.3) -- (\x,-8);
}
\draw[gray!40] (105,0.3) -- (105,-8);
\node[font=\scriptsize, anchor=east] at (105, 0.7) {13};
\node[font=\small\bfseries, anchor=east] at (-3,0.7) {2026};
\foreach \s/\e [count=\r from 0] in {0/15, 15/31, 31/51, 51/62, 62/82, 82/92, 92/99, 99/105} {
  \pgfmathsetmacro{\name}{\tasks[\r]}
  \node[font=\footnotesize, anchor=east] at (-3,-\r-0.5) {\name};
  \draw[bar] (\s,-\r-0.85) rectangle (\e,-\r-0.15);
}
\end{tikzpicture}
"""

DIAGRAMS = {"dfd0": DFD0, "dfd1": DFD1, "usecase": USECASE, "classes": CLASSES, "sequence": SEQUENCE,
            "plan": PLAN}


def render(name: str, body: str, out: Path) -> Path:
    """Compile one diagram and save it as a 300-dpi PNG."""
    work = out / "_tex" / name
    work.mkdir(parents=True, exist_ok=True)
    (work / "d.tex").write_text(PREAMBLE + body + "\n\\end{document}\n")
    subprocess.run(["tectonic", "d.tex"], cwd=work, check=True, capture_output=True)
    png = out / f"{name}.png"
    subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", "d.pdf", str(png.with_suffix(""))], cwd=work, check=True)
    return png


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "report/build/diagrams")
    out.mkdir(parents=True, exist_ok=True)
    for n, b in DIAGRAMS.items():
        print(render(n, b, out))
