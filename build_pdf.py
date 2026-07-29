"""
build_pdf.py -- render PAPER.md into a polished TWO-COLUMN conference-style PDF.

It extracts the title/author/abstract into a real title block, renders every
Markdown table as a full-width captioned float (table*), and places each figure
as a numbered full-width float (figure*) at its first reference, then compiles
with pandoc + tectonic. PAPER.md is untouched; paper_build.md / header_pdf.tex
are throwaways.

    python3 build_pdf.py        # needs pandoc + tectonic on PATH
"""

import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "PAPER.md")
BUILD = os.path.join(HERE, "paper_build.md")
HEADER = os.path.join(HERE, "header_pdf.tex")
OUT = os.path.join(HERE, "PAPER.pdf")

# (filename, display number, width-fraction of \textwidth, caption)
FIGURES = [
    ("figure9_detection_vs_monitoring.png", 6, 0.82, "The paper's central claim in one figure. AI-text detection operates on a single document; collapse monitoring operates on the distribution. Per-document statistics flag individual AI documents but miss collapse, while distribution-level statistics (vocabulary, n-gram diversity) detect collapse but say little about a single document."),
    ("figure10_collapse_scaling.png", 5, 0.96, "Collapse at scale (RAID seed; HC3 is similar). Left: trigram diversity falls at every per-generation corpus size (150 / 1{,}500 / 15{,}000 documents); larger corpora collapse more slowly but always collapse. Right: the per-document detector score stays flat and far below its threshold (dashed) at every size. The detection-vs-monitoring distinction is invariant to two orders of magnitude of scale."),
    ("figure5_roc_real_data.png", 1, 0.66, "ROC on HC3 (real human vs.\\ ChatGPT): SyntheticTextProbe against the perplexity and word-length baselines."),
    ("figure6_crossmodel.png", 2, 0.80, "Cross-model detection on RAID: one fixed zero-shot detector evaluated against each of eight generators."),
    ("figure7_lomo.png", 3, 0.88, "Leave-one-model-out: AUC on the held-out (unseen) generator for the trained classifier, perplexity alone, and the zero-shot heuristic."),
    ("figure8_neural_collapse.png", 4, 0.99, "Neural collapse (distilGPT-2 retrained on its own output). Left: diversity and detector score per generation. Right: perplexity of generated text under a fixed reference model, 69 to 7."),
]

TABLE_CAPTIONS = [
    "SyntheticTextProbe signals: weight, what each measures, and orientation.",
    "HC3 single-model held-out detection (positive class: ChatGPT).",
    "Cross-model zero-shot detection on RAID, per generator. Perplexity is re-fit in-distribution and shown for reference only.",
    "Leave-one-model-out AUC on the unseen generator (six signals + bigram perplexity). Reference seed 1234; 10-seed mean classifier AUC = 0.915 $\\pm$ 0.006. The zero-shot column is recomputed on the LOMO held-out human partition and therefore differs from Table 3.",
    "LOMO feature importance: mean absolute standardized coefficient across eight folds (bigram vs.\\ GPT-2 perplexity feature).",
    "Collapse endpoints: distribution metrics vs.\\ per-document detector score.",
    "Zero-shot AUC: GLTR under GPT-2 vs.\\ SyntheticTextProbe, computed on the full HC3/RAID pools (probe values therefore differ slightly from Tables 2--3).",
]

GLYPH_FIXES = {"⇒": "$\\Rightarrow$", "×": "$\\times$"}

HEADER_TEX = r"""
% --- conference-style polish (engine-agnostic; safe under XeTeX/tectonic) ---
\providecommand{\real}[1]{#1}     % pandoc 3.9 emits \real{} but doesn't define it
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{adjustbox}   % scale a too-wide table down to \textwidth
\usepackage{caption}
\captionsetup{font=small,labelfont=bf,skip=4pt}
\usepackage{titlesec}
\titleformat*{\section}{\large\bfseries}
\titleformat*{\subsection}{\normalsize\bfseries}
\titleformat*{\subsubsection}{\normalsize\itshape}
\setlength{\columnsep}{0.30in}
\setlength{\parindent}{1.2em}
% let full-width floats fill pages so 17 of them typeset cleanly
\extrafloats{60}
\renewcommand{\topfraction}{0.97}
\renewcommand{\bottomfraction}{0.95}
\renewcommand{\textfraction}{0.04}
\renewcommand{\floatpagefraction}{0.85}
\setcounter{topnumber}{4}
\setcounter{totalnumber}{6}
\widowpenalty=10000
\clubpenalty=10000
% absorb minor overfull lines in narrow two-column justified text
\setlength{\emergencystretch}{3em}
\tolerance=1200
\hbadness=10000
"""


def latex_cell(s):
    s = s.strip()
    s = s.replace("\\|", "|")
    for a, b in (("&", "\\&"), ("%", "\\%"), ("#", "\\#"), ("_", "\\_"),
                 ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}")):
        s = s.replace(a, b)
    s = s.replace("|", "\\textbar{}")
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"`(.+?)`", r"\\texttt{\1}", s)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\\emph{\1}", s)
    return s


def split_row(r):
    r = r.strip()
    if r.startswith("|"):
        r = r[1:]
    if r.endswith("|"):
        r = r[:-1]
    return [p.strip() for p in re.split(r"(?<!\\)\|", r)]


def is_numeric(x):
    x = x.replace("**", "").replace("*", "").strip()
    return bool(x) and bool(re.fullmatch(r"[-+]?\d[\d.,%/\sxn–\-]*\$?\\?[a-zA-Z]*\$?", x))


def render_table(block, tnum, caption):
    header = split_row(block[0])
    rows = [split_row(r) for r in block[2:] if r.strip()]
    ncol = len(header)
    aligns = []
    for c in range(ncol):
        col = [rows[i][c] for i in range(len(rows)) if c < len(rows[i])]
        aligns.append("r" if col and all(is_numeric(x) for x in col) else "l")
    if aligns:
        aligns[0] = "l"
    colspec = "@{}" + "".join(aligns) + "@{}"
    out = [
        "\\begin{table*}[t]\\centering",
        "\\caption*{\\textbf{Table %d:} %s}" % (tnum, caption),
        "\\small",
        "\\begin{adjustbox}{max width=\\textwidth}",
        "\\begin{tabular}{%s}" % colspec,
        "\\toprule",
        " & ".join("\\textbf{%s}" % latex_cell(h) for h in header) + " \\\\",
        "\\midrule",
    ]
    for r in rows:
        out.append(" & ".join(latex_cell(r[c]) if c < len(r) else "" for c in range(ncol)) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table*}", ""]
    return out


def fig_float(fname, num, width, caption):
    env, w = "figure*", "%.2f\\textwidth" % width
    return [
        "\\begin{%s}[t]\\centering" % env,
        "\\includegraphics[width=%s]{%s}" % (w, fname),
        "\\caption*{\\textbf{Figure %d:} %s}" % (num, caption),
        "\\end{%s}" % env,
        "",
    ]


def main():
    raw = open(SRC, encoding="utf-8").read().split("\n")

    # ---- 1. extract title / abstract, find body start ----
    title = raw[0].lstrip("# ").strip()
    abs_idx = next(i for i, l in enumerate(raw) if l.strip() == "## Abstract")
    abstract = ""
    for l in raw[abs_idx + 1:]:
        if l.strip() and l.strip() != "---":
            abstract = l.strip()
            break
    body_start = next(i for i, l in enumerate(raw) if l.strip().startswith("## 1."))
    body = raw[body_start:]

    # ---- 2. replace markdown tables with table* floats ----
    out, i, tnum = [], 0, 0
    while i < len(body):
        line = body[i]
        is_tbl_head = (line.strip().startswith("|")
                       and i + 1 < len(body)
                       and set(body[i + 1].strip()) <= set("|-: "))
        if is_tbl_head:
            block = []
            while i < len(body) and body[i].strip().startswith("|"):
                block.append(body[i])
                i += 1
            cap = TABLE_CAPTIONS[tnum] if tnum < len(TABLE_CAPTIONS) else ""
            tnum += 1
            out += render_table(block, tnum, cap)
        elif re.match(r"^\*\*Table \d+\.\*\*", line.strip()):
            i += 1
        else:
            out.append(line)
            i += 1
    body = out

    # ---- 3. strip printed figure filenames, then insert figure* floats ----
    fig_first = {}
    for fname, num, _w, _c in FIGURES:
        for idx, l in enumerate(body):
            if fname in l:
                fig_first[num] = idx
                break
    # remove the "(*figureN_...png*)" / "*figureN_...png*" tokens from prose
    for idx in range(len(body)):
        if "\\includegraphics" in body[idx]:
            continue
        body[idx] = re.sub(r"\s*\(\*figure\d+_[a-z0-9_]+\.png\*\)", "", body[idx])
        body[idx] = re.sub(r"\*figure\d+_[a-z0-9_]+\.png\*", "", body[idx])
        body[idx] = body[idx].replace(", )", ")").replace("( )", "").replace(" )", ")")
        body[idx] = re.sub(r"\*\*\s*:", "**:", body[idx])
    # insert floats bottom-up so indices stay valid
    inserts = sorted(((idx, num) for num, idx in fig_first.items()), key=lambda x: -x[0])
    fmap = {num: (fname, w, c) for fname, num, w, c in FIGURES}
    for idx, num in inserts:
        fname, w, c = fmap[num]
        for ln in reversed(fig_float(fname, num, w, c)):
            body.insert(idx + 1, ln)

    text = "\n".join(body)
    for bad, good in GLYPH_FIXES.items():
        text = text.replace(bad, good)

    # ---- 4. YAML metadata title block + body ----
    meta = (
        "---\n"
        "title: |\n  " + title + "\n"
        "author: \"Shaheer Saud, *Independent*, shaheersaud2004@gmail.com\"\n"
        "date: \"\"\n"
        "abstract: |\n  " + abstract + "\n"
        "---\n\n"
    )
    open(BUILD, "w", encoding="utf-8").write(meta + text)
    open(HEADER, "w", encoding="utf-8").write(HEADER_TEX)
    print("wrote", BUILD)

    cmd = [
        "pandoc", BUILD, "-o", OUT,
        "--pdf-engine=tectonic",
        "-f", "markdown",
        "-H", HEADER,
        "-V", "classoption=twocolumn",
        "-V", "documentclass=article",
        "-V", "geometry:margin=0.78in",
        "-V", "fontsize=10pt",
        "-V", "colorlinks=true", "-V", "linkcolor=RoyalBlue", "-V", "urlcolor=RoyalBlue",
    ]
    print("running pandoc + tectonic ...")
    if subprocess.run(cmd, cwd=HERE).returncode != 0:
        print("pandoc failed")
        sys.exit(1)
    print("OK ->", OUT)


if __name__ == "__main__":
    main()
