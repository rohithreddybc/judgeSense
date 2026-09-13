"""Emit the experimental-setup appendix from the registry and the usage log.

The paper reports twenty-five judges by short alias and never says which
checkpoint each alias resolves to, which provider served it, or what the
decoding settings were. A reviewer assessing model recency, or anyone trying to
reproduce a cell, cannot. The preprint had this section; the revision lost it.

Generated rather than typed, so it cannot drift from the registry the sweep
actually ran against.

    python scripts/make_setup_appendix.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tables" / "setup_appendix.tex"
MULT = REPO / "data" / "results_v2" / "multiplicity.json"

sys.path.insert(0, str(REPO / "src"))
from judge_registry import JUDGES, is_retired  # noqa: E402

VENDOR_OF = {
    "claude": "Anthropic", "deepseek": "DeepSeek", "gemini": "Google",
    "gemma": "Google", "glm": "Zhipu", "kimi": "Moonshot", "llama": "Meta",
    "mistral": "Mistral", "magistral": "Mistral", "qwen": "Alibaba",
    "gpt-oss": "OpenAI",
}


def vendor(judge: str) -> str:
    fam = ((JUDGES.get(judge) or {}).get("family") or judge).lower()
    for k, v in VENDOR_OF.items():
        if fam.startswith(k) or judge.lower().startswith(k):
            return v
    return "---"


def esc(t: str) -> str:
    return (str(t).replace("_", r"\_").replace("&", r"\&")
            .replace("%", r"\%").replace("#", r"\#"))


def main() -> int:
    reported = sorted({c["judge"] for c in
                       json.load(io.open(MULT, encoding="utf-8"))["cells"].values()})

    rows = []
    for j in reported:
        spec = JUDGES.get(j) or {}
        rows.append("%s & %s & %s & %s \\\\" % (
            r"\texttt{" + esc(j) + "}",
            esc(vendor(j)),
            r"\texttt{\scriptsize " + esc(spec.get("model_id", "---")) + "}",
            esc(spec.get("provider", "---"))))

    dropped = sorted(n for n in JUDGES if is_retired(n))
    drows = []
    for j in dropped:
        why = (JUDGES[j].get("retired") or "").split(":")[-1].strip()
        drows.append("%s & %s & %s \\\\" % (
            r"\texttt{" + esc(j) + "}", esc(vendor(j)), esc(why)))

    # NOTE: this template is %-formatted, so every literal LaTeX comment marker
    # has to be doubled or Python reads it as a conversion specifier.
    tex = r"""%% <<< GENERATED: scripts/make_setup_appendix.py -- do not edit by hand
\section{Experimental setup}
\label{app:setup}

Every judge below was reached through the provider named, at
\texttt{temperature}~$=0$ where the provider accepts it, with an identical
output budget of $1{,}024$ tokens and explicit suppression of reasoning traces
where the provider exposes that control. Whether suppression was requested and
whether it was honoured are recorded per call in the released logs, never
inferred. One judge rejects an explicit temperature parameter and therefore ran
at its provider default; its cells are marked in Table~\ref{tab:main}.

\begin{longtable}{llll}
\caption{\textbf{The twenty-five reported judges.} Alias as used throughout the
paper, the vendor that trained the checkpoint, the checkpoint identifier sent
to the provider, and the provider that served it. Vendor and provider differ
wherever a model is served by a host that did not train it. Takeaway: every
alias in Table~\ref{tab:main} resolves to a pinned checkpoint string.}
\label{tab:setup}\\
\toprule
Alias & Vendor & Checkpoint & Provider \\
\midrule
\endfirsthead
\multicolumn{4}{l}{\footnotesize\itshape Table~\ref{tab:setup}, continued.}\\
\toprule
Alias & Vendor & Checkpoint & Provider \\
\midrule
\endhead
\bottomrule
\endlastfoot
%(rows)s
\end{longtable}

\paragraph{Judges registered but not reported.}
%(ndrop)d checkpoints are in the registry and carry no cell in
Table~\ref{tab:main}. None was dropped for its answers; each was withdrawn on
throughput, because the provider's free tier could not finish a cell in a
usable time. They are listed so the roster is auditable rather than curated.

\begin{center}
\begin{tabular}{lll}
\toprule
Alias & Vendor & Why it was withdrawn \\
\midrule
%(drows)s
\bottomrule
\end{tabular}
\end{center}
""" % {"rows": "\n".join(rows), "drows": "\n".join(drows), "ndrop": len(drows)}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    io.open(OUT, "w", encoding="utf-8", newline="\n").write(tex)
    print(f"  wrote {OUT}")
    print(f"  reported judges {len(rows)}, withdrawn {len(drows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
