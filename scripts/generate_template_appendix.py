"""Emit the paraphrase-template appendix from the builder's own TEMPLATES dict.

The templates are the manipulation the whole benchmark rests on, so a reader
cannot check the equivalence claim without seeing them. Generated rather than
transcribed, and spliced between sentinels, so the appendix cannot drift from
the strings the dataset was actually built with -- the same discipline the
results table uses.

Run:  python scripts/generate_template_appendix.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dataset_builder_v2 import TEMPLATES  # noqa: E402

MANUSCRIPT = REPO / "paper" / "main.tex"
BEGIN = "% <<< GENERATED APPENDIX: generate_template_appendix.py -- do not edit by hand"
END = "% >>> END GENERATED APPENDIX"

TASK_TITLE = {
    "factuality": "Factuality (TruthfulQA)",
    "coherence": "Coherence (SummEval)",
    "relevance": "Relevance (BEIR TREC-COVID)",
    "preference": "Preference (MT-Bench human judgments)",
}
ANSWER_SPACE = {
    "factuality": r"\{YES, NO\}",
    "coherence": r"$\{1,\dots,5\}$",
    "relevance": r"\{A, B\}",
    "preference": r"\{A, B\}",
}


def _tex_escape(s: str) -> str:
    """Escape for verbatim-ish display inside \\texttt."""
    for a, b in (("\\", r"\textbackslash{}"), ("{", r"\{"), ("}", r"\}"),
                 ("$", r"\$"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"),
                 ("_", r"\_"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def build() -> str:
    out = [
        r"\appendix",
        r"\section{The twenty instruction templates}",
        r"\label{app:templates}",
        "",
        r"Every template shipped with the benchmark is printed below, exactly as",
        r"the builder emits it. Placeholders in braces are filled per item; the",
        r"payload is otherwise unchanged between the two arms of a pair.",
        "",
        r"Each item is issued under exactly two of the five templates for its",
        r"task, assigned by a label-stratified seeded permutation over the ten",
        r"unordered pairs (Section~\ref{sec:benchmark}). No item appears under",
        r"more than one pair, so template identity is nested inside item identity",
        r"and no between-template contrast is identifiable from this design.",
        "",
        r"The audit described in Section~\ref{sec:benchmark} is run against these",
        r"strings: it checks that the requested answer set is identical within",
        r"each pair, that no template inverts polarity, that each names its own",
        r"task's construct, and that no pair is a near-duplicate. What it cannot",
        r"establish is that a competent reader maps two wordings onto the same",
        r"question, which is why the equivalence claim is stated as an assumption",
        r"and not as a result.",
        "",
    ]
    for task in ("factuality", "coherence", "relevance", "preference"):
        templates = TEMPLATES[task]
        out += [
            rf"\subsection{{{TASK_TITLE[task]}}}",
            rf"Requested answer set: {ANSWER_SPACE[task]} for all five templates.",
            "",
        ]
        for i, tpl in enumerate(templates, start=1):
            body = _tex_escape(tpl).replace("\\n\\n", " ")
            body = " ".join(body.split())
            # \sloppy + \raggedright: these are long unhyphenated monospace
            # strings, and a justified \texttt line cannot break them without
            # overfull boxes. Set as a block rather than a description item so
            # the text has the full measure to break across.
            out += [
                rf"\paragraph{{T{i}.}}",
                r"{\ttfamily\small\raggedright\sloppy",
                f"  {body}",
                r"\par}",
                "",
            ]
    return "\n".join(out).rstrip()


def main() -> int:
    block = f"{BEGIN}\n{build()}\n{END}"
    text = MANUSCRIPT.read_text(encoding="utf-8")
    start, end = text.find(BEGIN), text.find(END)
    if start != -1 and end != -1:
        updated = text[:start] + block + text[end + len(END):]
    else:
        # first insertion: immediately before \bibliographystyle
        anchor = text.find(r"\bibliographystyle")
        if anchor == -1:
            print("could not find \\bibliographystyle; appendix not inserted")
            return 1
        updated = text[:anchor] + block + "\n\n" + text[anchor:]
    if updated != text:
        MANUSCRIPT.write_text(updated, encoding="utf-8")
        print(f"  spliced appendix: {sum(len(v) for v in TEMPLATES.values())} templates "
              f"across {len(TEMPLATES)} tasks")
    else:
        print("  appendix already current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
