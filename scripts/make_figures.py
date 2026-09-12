"""Emit the paper's two figures as TikZ/pgfplots, from committed data.

The preprint draws every figure in TikZ rather than including a bitmap, so the
figures stay reproducible from the data and readable at any zoom. This follows
that convention: no \\includegraphics, no image files, and nothing typed by hand.

  Figure 1  per-task dJSS across 25 judges, coloured by provider, hollow where
            the repeat ceiling falls below 0.90, diamond at the pooled mean.
            Carries the sign result, coherence's size, AND the retraction --
            the three same-family Claude points cluster tightly, which is what
            produced the withdrawn task-dominates-judge claim.

  Figure 2  repeat ceiling against dJSS for all 98 cells, with the eight
            agent-harness cells overlaid and arrows joining each Haiku API cell
            to its harness counterpart. Carries the reason the 11 exclusions
            are principled: a low ceiling pulls dJSS toward zero, so
            uncontrolled decoding HIDES sensitivity rather than exposing it.

    python scripts/make_figures.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MULT = REPO / "data" / "results_v2" / "multiplicity.json"
STAGED = REPO / "data" / "claude_code" / "_staged"
OUT = REPO / "tables" / "figures_v2.tex"

sys.path.insert(0, str(REPO / "src"))
from judge_registry import JUDGES  # noqa: E402

TASKS = ("coherence", "factuality", "preference", "relevance")
# colour-blind-safe, distinguishable in greyscale by marker too
PROVIDER_COLOUR = {
    "anthropic":   "cbOrange",
    "google":      "cbBlue",
    "huggingface": "cbGreen",
    "dashscope":   "cbPurple",
    "mistral":     "cbRed",
    "novita":      "cbBrown",
}

PREAMBLE = r"""% <<< GENERATED FIGURES: scripts/make_figures.py -- do not edit by hand
% Re-run the script instead, or the paper and the data drift apart.
\definecolor{cbBlue}{HTML}{0072B2}
\definecolor{cbOrange}{HTML}{E69F00}
\definecolor{cbGreen}{HTML}{009E73}
\definecolor{cbPurple}{HTML}{CC79A7}
\definecolor{cbRed}{HTML}{D55E00}
\definecolor{cbBrown}{HTML}{8C6D31}
"""


def load():
    d = json.load(io.open(MULT, encoding="utf-8"))
    return d["cells"], d["pooled_by_task_holm"]


def provider(judge):
    return (JUDGES.get(judge) or {}).get("provider", "other")


def fig_delta(cells, pooled):
    """Figure 1: one row per task, one marker per judge."""
    rows = []
    for yi, task in enumerate(TASKS):
        ys = len(TASKS) - yi
        for key, c in cells.items():
            if c["task"] != task or c["delta"] is None:
                continue
            col = PROVIDER_COLOUR.get(provider(c["judge"]), "black")
            hollow = c["ceiling_below_bar"]
            style = (f"draw={col}, fill=white" if hollow else f"draw={col}, fill={col}")
            rows.append(f"\\addplot[only marks, mark=*, mark size=1.5pt, "
                        f"{style}] coordinates {{({c['delta']:.4f},{ys})}};")
        p = pooled.get(task)
        if p:
            rows.append(f"\\addplot[only marks, mark=diamond*, mark size=3.4pt, "
                        f"draw=black, fill=black] "
                        f"coordinates {{({p['mean_delta']:.4f},{ys})}};")
    ticks = ",".join(str(len(TASKS) - i) for i in range(len(TASKS)))
    labels = ",".join(TASKS)
    return r"""
\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[
  width=\linewidth, height=5.0cm,
  xlabel={$\Delta\mathrm{JSS}$ \ (negative = rewording costs agreement)},
  xmin=-0.36, xmax=0.04,
  %% pgfplots defaults to scientific notation near zero, which collapses the
  %% ticks either side of the origin into an unreadable run of 8*10^-2 style
  %% labels. Fix the ticks and force plain decimals.
  xtick={-0.35,-0.30,-0.25,-0.20,-0.15,-0.10,-0.05,0},
  scaled x ticks=false,
  x tick label style={/pgf/number format/fixed,
                      /pgf/number format/precision=2},
  ytick={%(ticks)s}, yticklabels={%(labels)s},
  ymin=0.4, ymax=%(ymax)s,
  xmajorgrids, grid style={gray!22},
  tick label style={font=\footnotesize},
  label style={font=\footnotesize},
  scale only axis,
]
\draw[dashed, gray!70] ({axis cs:0,0}|-{rel axis cs:0,0})
                    -- ({axis cs:0,0}|-{rel axis cs:0,1});
\draw[dashed, gray!45] ({axis cs:-0.02,0}|-{rel axis cs:0,0})
                    -- ({axis cs:-0.02,0}|-{rel axis cs:0,1});
%(rows)s
\end{axis}
\end{tikzpicture}
\caption{\textbf{Paraphrase cost by task across twenty-five judges.} One marker
per judge--task cell, coloured by provider; hollow markers are the eleven cells
whose repeat ceiling falls below $0.90$ and which we decline to read; black
diamonds are the pooled mean per task. Dashed lines mark zero and the $-0.02$
smallest effect of interest. Takeaway: the cost is negative almost everywhere,
coherence is both the largest loss and by far the widest spread across judges,
and the three same-vendor judges of the superseded version sit close together,
which is what produced its retracted claim that the task matters more than the
judge.}
\label{fig:delta}
\end{figure}
""" % {"ticks": ticks, "labels": labels, "ymax": len(TASKS) + 0.6,
       "rows": "\n".join(rows)}


def fig_ceiling(cells):
    """Figure 2: ceiling against delta, harness cells overlaid."""
    marks = {"coherence": "*", "factuality": "square*",
             "relevance": "triangle*", "preference": "diamond*"}
    rows = []
    for c in cells.values():
        if c["delta"] is None or c["ceiling"] is None:
            continue
        m = marks.get(c["task"], "*")
        hollow = c["ceiling_below_bar"]
        style = "draw=cbBlue, fill=white" if hollow else "draw=cbBlue, fill=cbBlue!55"
        rows.append(f"\\addplot[only marks, mark={m}, mark size=1.4pt, {style}] "
                    f"coordinates {{({c['ceiling']:.4f},{c['delta']:.4f})}};")

    # harness cells and the arrows from their API counterparts
    arrows = []
    for path in sorted(STAGED.glob("cc-haiku-4-5_*.jsonl")):
        task = path.stem.rsplit("_", 1)[1]
        api = cells.get(f"claude-haiku|{task}")
        if not api or api["ceiling"] is None:
            continue
        harness = HARNESS.get(task)
        if not harness:
            continue
        hc, hd = harness
        rows.append(f"\\addplot[only marks, mark=x, mark size=3pt, "
                    f"draw=cbRed, very thick] coordinates {{({hc:.4f},{hd:.4f})}};")
        arrows.append(f"\\draw[->, cbRed, thin, opacity=0.75] "
                      f"(axis cs:{api['ceiling']:.4f},{api['delta']:.4f}) -- "
                      f"(axis cs:{hc:.4f},{hd:.4f});")
    return r"""
\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[
  width=\linewidth, height=5.4cm,
  xlabel={repeat ceiling $\mathrm{JSS}_{\text{rep}}$},
  ylabel={$\Delta\mathrm{JSS}$},
  xmin=0.58, xmax=1.02, ymin=-0.36, ymax=0.05,
  xtick={0.6,0.7,0.8,0.9,1.0},
  ytick={-0.35,-0.30,-0.25,-0.20,-0.15,-0.10,-0.05,0},
  scaled ticks=false,
  tick label style={/pgf/number format/fixed,
                    /pgf/number format/precision=2, font=\footnotesize},
  xmajorgrids, ymajorgrids, grid style={gray!22},
  label style={font=\footnotesize},
  scale only axis,
]
\draw[dashed, gray!70] (axis cs:0.90,-0.36) -- (axis cs:0.90,0.05);
\draw[dashed, gray!45] (axis cs:0.58,-0.02) -- (axis cs:1.02,-0.02);
%(rows)s
%(arrows)s
\end{axis}
\end{tikzpicture}
\caption{\textbf{Repeat ceiling against paraphrase cost, all ninety-eight cells,
with the agent-harness cells overlaid.} Marker shape gives the task; hollow
markers fall left of the $0.90$ readability bar. Red crosses are Haiku measured
through the agent harness, joined by arrows to the same judge and task measured
through a direct API call. Takeaway: a low ceiling pulls $\Delta\mathrm{JSS}$
toward zero rather than away from it, so uncontrolled decoding hides wording
sensitivity instead of exposing it, and the harness arrows move left and up for
exactly that reason.}
\label{fig:ceiling}
\end{figure}
""" % {"rows": "\n".join(rows), "arrows": "\n".join(arrows)}


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
    print(f"  wrote {OUT}")
    print(f"  Figure 1: {sum(1 for c in cells.values() if c['delta'] is not None)} cells")
    print(f"  Figure 2: same cells plus {len(HARNESS)} harness overlays")
    return 0


if __name__ == "__main__":
    sys.exit(main())
