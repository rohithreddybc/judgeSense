"""Emit the paper's data figures as TikZ/pgfplots, from committed data.

Everything is drawn in vector TikZ rather than included as a bitmap, so the
figures regenerate from the data, stay sharp at any zoom, and inherit the
document's fonts.

  Figure 2  per-task dJSS across 25 judges, coloured by VENDOR, hollow where
            the repeat ceiling falls below 0.90, diamond at the pooled mean.
            Carries the sign result, coherence's size, and the retraction: the
            three same-vendor Anthropic points cluster tightly, which is what
            produced the withdrawn task-dominates-judge claim.

  Figure 3  repeat ceiling against dJSS for all 98 cells, with the eight
            agent-harness cells overlaid and arrows joining each Haiku API cell
            to its harness counterpart. Carries the reason the 11 exclusions
            are principled: a low ceiling pulls dJSS toward zero, so
            uncontrolled decoding HIDES sensitivity rather than exposing it.

TWO THINGS THIS FIXES

Neither figure had a legend, while both captions promised an encoding ("one
marker per judge-task cell, coloured by provider"; "marker shape gives the
task"). A reader could not decode either one.

And the colouring was by PROVIDER, the host an endpoint was called through,
while the analysis clusters by VENDOR, whoever trained the checkpoint. Those
differ: llama-4-scout is served by huggingface and trained by Meta. The figure
now matches the inference.

    python scripts/make_figures.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MULT = REPO / "data" / "results_v2" / "multiplicity.json"
OUT = REPO / "tables" / "figures_v2.tex"

sys.path.insert(0, str(REPO / "src"))
from judge_registry import JUDGES  # noqa: E402

TASKS = ("coherence", "factuality", "preference", "relevance")

# Whoever trained the checkpoint, not whoever serves it. Matches the clustering
# unit in the confirmatory contrast, so the picture and the test agree.
VENDOR_OF = {
    "claude": "Anthropic", "deepseek": "DeepSeek", "gemini": "Google",
    "gemma": "Google", "glm": "Zhipu", "kimi": "Moonshot", "llama": "Meta",
    "mistral": "Mistral", "magistral": "Mistral", "qwen": "Alibaba",
}
# Okabe-Ito, colour-blind safe; shape and fill carry the same information for
# greyscale.
VENDOR_COLOUR = {
    "Alibaba": "vAlibaba", "Meta": "vMeta", "Anthropic": "vAnthropic",
    "DeepSeek": "vDeepSeek", "Google": "vGoogle", "Mistral": "vMistral",
    "Zhipu": "vZhipu", "Moonshot": "vMoonshot",
}
TASK_MARK = {"coherence": "*", "factuality": "square*",
             "relevance": "triangle*", "preference": "diamond*"}

PREAMBLE = r"""% <<< GENERATED FIGURES: scripts/make_figures.py -- do not edit by hand
% Re-run the script instead, or the paper and the data drift apart.
\definecolor{vAlibaba}{HTML}{0072B2}
\definecolor{vMeta}{HTML}{D55E00}
\definecolor{vAnthropic}{HTML}{CC79A7}
\definecolor{vDeepSeek}{HTML}{009E73}
\definecolor{vGoogle}{HTML}{E69F00}
\definecolor{vMistral}{HTML}{56B4E9}
\definecolor{vZhipu}{HTML}{8C6D31}
\definecolor{vMoonshot}{HTML}{555555}
\definecolor{gridGrey}{HTML}{D8D8D8}
\definecolor{ruleGrey}{HTML}{8A8A8A}
"""


def vendor(judge):
    fam = ((JUDGES.get(judge) or {}).get("family") or judge).lower()
    for key, v in VENDOR_OF.items():
        if fam.startswith(key) or judge.lower().startswith(key):
            return v
    return "Other"


def load():
    d = json.load(io.open(MULT, encoding="utf-8"))
    return d["cells"], d["pooled_by_task_holm"]


def fig_delta(cells, pooled):
    rows, seen = [], set()
    for yi, task in enumerate(TASKS):
        ys = len(TASKS) - yi
        for c in cells.values():
            if c["task"] != task or c["delta"] is None:
                continue
            v = vendor(c["judge"])
            col = VENDOR_COLOUR.get(v, "black")
            style = (f"draw={col}, fill=white, line width=0.7pt"
                     if c["ceiling_below_bar"] else f"draw={col}, fill={col}")
            rows.append(f"\\addplot[only marks, mark=*, mark size=1.7pt, "
                        f"{style}, forget plot] "
                        f"coordinates {{({c['delta']:.4f},{ys})}};")
            seen.add(v)
        p = pooled.get(task)
        if p:
            rows.append(f"\\addplot[only marks, mark=diamond*, mark size=3.6pt, "
                        f"draw=black, fill=black, forget plot] "
                        f"coordinates {{({p['mean_delta']:.4f},{ys})}};")

    # \addlegendimage draws a key with no data point behind it. Plotting a
    # phantom point off-canvas instead overflows pgfplots' coordinate
    # transform under clip=false ("Dimension too large").
    legend = []
    for v in sorted(seen):
        col = VENDOR_COLOUR.get(v, "black")
        legend.append(f"\\addlegendimage{{only marks, mark=*, mark size=1.7pt, "
                      f"draw={col}, fill={col}}}")
        legend.append(f"\\addlegendentry{{{v}}}")
    legend.append("\\addlegendimage{only marks, mark=*, mark size=1.7pt, "
                  "draw=black, fill=white, line width=0.7pt}")
    legend.append("\\addlegendentry{ceiling $<0.90$}")
    legend.append("\\addlegendimage{only marks, mark=diamond*, mark size=3.6pt, "
                  "draw=black, fill=black}")
    legend.append("\\addlegendentry{pooled mean}")

    ticks = ",".join(str(len(TASKS) - i) for i in range(len(TASKS)))
    return r"""
\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[
  width=\linewidth, height=5.2cm,
  xlabel={$\Delta\mathrm{JSS}$\ \ (negative: rewording costs agreement)},
  xmin=-0.37, xmax=0.055,
  xtick={-0.35,-0.30,-0.25,-0.20,-0.15,-0.10,-0.05,0},
  scaled x ticks=false,
  x tick label style={/pgf/number format/fixed, /pgf/number format/precision=2},
  ytick={%(ticks)s}, yticklabels={%(labels)s},
  ymin=0.45, ymax=%(ymax)s,
  xmajorgrids, grid style={gridGrey, line width=0.4pt},
  axis line style={ruleGrey}, tick style={ruleGrey},
  tick label style={font=\footnotesize},
  label style={font=\footnotesize},
  legend style={font=\scriptsize, draw=none, fill=none,
                at={(0.5,-0.30)}, anchor=north, legend columns=5,
                cells={anchor=west}, column sep=7pt, inner sep=1pt},
  legend image post style={scale=0.85},
  clip=false, scale only axis,
]
\draw[ruleGrey, line width=0.5pt] ({axis cs:0,0}|-{rel axis cs:0,0})
                               -- ({axis cs:0,0}|-{rel axis cs:0,1});
\draw[dashed, ruleGrey] ({axis cs:-0.02,0}|-{rel axis cs:0,0})
                     -- ({axis cs:-0.02,0}|-{rel axis cs:0,1});
%(rows)s
%(legend)s
\end{axis}
\end{tikzpicture}
\caption{\textbf{Paraphrase cost by task across twenty-five judges.} One marker
per judge--task cell, coloured by the vendor that trained the checkpoint, which
is also the unit the confirmatory contrast clusters on. Hollow markers are the
eleven cells whose repeat ceiling falls below $0.90$ and which we decline to
read. Black diamonds are the pooled mean per task; the solid rule is zero and
the dashed rule the $-0.02$ smallest effect of interest. Takeaway: the cost is
negative almost everywhere, coherence is both the largest loss and by far the
widest spread across judges, and the three Anthropic judges of the superseded
version sit close together, which is what produced its retracted claim that the
task matters more than the judge.}
\label{fig:delta}
\end{figure}
""" % {"ticks": ticks, "labels": ",".join(TASKS), "ymax": len(TASKS) + 0.55,
       "rows": "\n".join(rows), "legend": "\n".join(legend)}


def fig_ceiling(cells):
    rows, seen = [], set()
    for c in cells.values():
        if c["delta"] is None or c["ceiling"] is None:
            continue
        m = TASK_MARK.get(c["task"], "*")
        style = ("draw=vAlibaba, fill=white, line width=0.7pt"
                 if c["ceiling_below_bar"] else "draw=vAlibaba, fill=vAlibaba!45")
        rows.append(f"\\addplot[only marks, mark={m}, mark size=1.6pt, {style}, "
                    f"forget plot] "
                    f"coordinates {{({c['ceiling']:.4f},{c['delta']:.4f})}};")
        seen.add(c["task"])

    arrows = []
    for task, (hc, hd) in HARNESS.items():
        api = cells.get(f"claude-haiku|{task}")
        if not api or api["ceiling"] is None:
            continue
        rows.append(f"\\addplot[only marks, mark=x, mark size=3.2pt, "
                    f"draw=vMeta, line width=1pt, forget plot] "
                    f"coordinates {{({hc:.4f},{hd:.4f})}};")
        arrows.append(f"\\draw[->, vMeta, line width=0.6pt, opacity=0.8] "
                      f"(axis cs:{api['ceiling']:.4f},{api['delta']:.4f}) -- "
                      f"(axis cs:{hc:.4f},{hd:.4f});")

    legend = []
    for t in TASKS:
        if t not in seen:
            continue
        legend.append(f"\\addlegendimage{{only marks, mark={TASK_MARK[t]}, "
                      f"mark size=1.6pt, draw=vAlibaba, fill=vAlibaba!45}}")
        legend.append(f"\\addlegendentry{{{t}}}")
    legend.append("\\addlegendimage{only marks, mark=*, mark size=1.6pt, "
                  "draw=vAlibaba, fill=white, line width=0.7pt}")
    legend.append("\\addlegendentry{ceiling $<0.90$}")
    legend.append("\\addlegendimage{only marks, mark=x, mark size=3.2pt, "
                  "draw=vMeta, line width=1pt}")
    legend.append("\\addlegendentry{agent harness}")

    return r"""
\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[
  width=\linewidth, height=5.6cm,
  xlabel={repeat ceiling $\mathrm{JSS}_{\text{rep}}$\ \ (the judge's agreement with itself)},
  ylabel={$\Delta\mathrm{JSS}$},
  xmin=0.58, xmax=1.025, ymin=-0.37, ymax=0.055,
  xtick={0.6,0.7,0.8,0.9,1.0},
  ytick={-0.35,-0.30,-0.25,-0.20,-0.15,-0.10,-0.05,0},
  scaled ticks=false,
  tick label style={/pgf/number format/fixed, /pgf/number format/precision=2,
                    font=\footnotesize},
  label style={font=\footnotesize},
  xmajorgrids, ymajorgrids, grid style={gridGrey, line width=0.4pt},
  axis line style={ruleGrey}, tick style={ruleGrey},
  legend style={font=\scriptsize, draw=none, fill=none,
                at={(0.5,-0.30)}, anchor=north, legend columns=5,
                cells={anchor=west}, column sep=7pt, inner sep=1pt},
  legend image post style={scale=0.85},
  clip=false, scale only axis,
]
\draw[dashed, ruleGrey, line width=0.6pt] (axis cs:0.90,-0.37) -- (axis cs:0.90,0.055);
\node[font=\scriptsize, text=ruleGrey, anchor=south east, inner sep=1.5pt]
  at (axis cs:0.895,-0.37) {not read};
\draw[dashed, ruleGrey] (axis cs:0.58,-0.02) -- (axis cs:1.025,-0.02);
%(rows)s
%(arrows)s
%(legend)s
\end{axis}
\end{tikzpicture}
\caption{\textbf{Repeat ceiling against paraphrase cost, all ninety-eight
cells, with the agent-harness cells overlaid.} Marker shape gives the task and
hollow markers fall left of the $0.90$ readability bar. Crosses are Haiku
measured through the agent harness, joined by arrows to the same judge and task
measured through a direct API call. Takeaway: a low ceiling pulls
$\Delta\mathrm{JSS}$ toward zero rather than away from it, so uncontrolled
decoding hides wording sensitivity instead of exposing it, and the harness
arrows move left and up for exactly that reason.}
\label{fig:ceiling}
\end{figure}
""" % {"rows": "\n".join(rows), "arrows": "\n".join(arrows),
       "legend": "\n".join(legend)}


# Haiku through the harness: (repeat ceiling, delta), from the transport control.
HARNESS = {
    "factuality": (0.910, -0.010),
    "coherence":  (0.684, -0.124),
    "relevance":  (0.829, -0.029),
    "preference": (0.881, -0.012),
}


def main() -> int:
    cells, pooled = load()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    io.open(OUT, "w", encoding="utf-8", newline="\n").write(
        PREAMBLE + fig_delta(cells, pooled) + fig_ceiling(cells))
    vendors = sorted({vendor(c["judge"]) for c in cells.values()})
    print(f"  wrote {OUT}")
    print(f"  vendors on the legend: {len(vendors)} -> {', '.join(vendors)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
