#!/usr/bin/env python3
"""
Task 5 — Publication figures for JudgeSense paper.

Outputs (all 300 dpi, serif font, min 11pt):
  outputs/fig1_coherence_jss_bar.pdf
  outputs/fig2_jss_heatmap.pdf
  outputs/fig4_factuality_raw_vs_corrected.pdf

fig3 is intentionally skipped (cluttered).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "data/results/raw_outputs"
OUTPUT_DIR  = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))
from src.metrics import judge_sensitivity_score, bootstrap_confidence_interval

matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
})

MODELS = [
    # Existing 8 (max_tokens=20)
    "claude-haiku", "claude-sonnet", "gemini-flash",
    "gpt-4o-mini", "gpt-4o", "llama3-70b", "mistral-7b", "qwen",
    # Re-run at max_tokens=1024 to retire the truncation caveat
    "deepseek",
    # 4 new judges (max_tokens=1024) added in revision pass 2
    "gpt-5.5", "claude-opus-4-7", "qwen-3.6-flash", "deepseek-v4-flash",
]

DISPLAY = {
    "claude-haiku":       "Claude Haiku",
    "claude-sonnet":      "Claude Sonnet",
    "deepseek":           "DeepSeek-R1",
    "gemini-flash":       "Gemini Flash",
    "gpt-4o-mini":        "GPT-4o-mini",
    "gpt-4o":             "GPT-4o",
    "llama3-70b":         "LLaMA3-70B",
    "mistral-7b":         "Mistral 7B",
    "qwen":               "Qwen 2.5-72B",
    # Pass-2 additions
    "gpt-5.5":            "GPT-5.5",
    "claude-opus-4-7":    "Claude Opus 4.7",
    "qwen-3.6-flash":     "Qwen 3.6 Flash",
    "deepseek-v4-flash":  "DeepSeek-V4 Flash",
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_task_jss(task: str) -> dict:
    """Return {model: (jss, norm_a_list, norm_b_list)} for the given task."""
    out = {}
    for model in MODELS:
        path = RESULTS_DIR / f"{model}_{task}.jsonl"
        if not path.exists():
            continue
        a_list, b_list = [], []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("error") is not None:
                    continue
                na = rec.get("normalized_a")
                nb = rec.get("normalized_b")
                if not na or not nb:
                    continue
                a_list.append(str(na))
                b_list.append(str(nb))
        if a_list:
            jss = judge_sensitivity_score(a_list, b_list)
            out[model] = (jss, a_list, b_list)
    return out


# ---------------------------------------------------------------------------
# Figure 1 — Coherence JSS horizontal bar chart
# ---------------------------------------------------------------------------

def fig1_coherence_bar():
    data = load_task_jss("coherence")

    rows = []
    for model in MODELS:
        if model not in data:
            continue
        jss, a_list, b_list = data[model]
        lo, hi = bootstrap_confidence_interval(a_list, b_list, judge_sensitivity_score)
        rows.append({
            "model": model,
            "display": DISPLAY[model],
            "jss": jss,
            "lo": lo,
            "hi": hi,
        })

    rows.sort(key=lambda r: r["jss"])  # ascending so highest is at top visually

    labels   = [r["display"] for r in rows]
    values   = [r["jss"]     for r in rows]
    xerr_lo  = [r["jss"] - r["lo"] for r in rows]
    xerr_hi  = [r["hi"] - r["jss"] for r in rows]

    colors = []
    for v in values:
        if v > 0.9:
            colors.append("#2ecc71")
        elif v >= 0.65:
            colors.append("#f39c12")
        else:
            colors.append("#e74c3c")

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, xerr=[xerr_lo, xerr_hi], color=colors,
            ecolor="black", capsize=3, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Judge Sensitivity Score")
    ax.set_title("Coherence Task: JSS by Model")
    ax.axvline(x=0.8, color="grey", linestyle="--", linewidth=1, label="JSS = 0.8")

    legend_patches = [
        mpatches.Patch(color="#2ecc71", label="JSS > 0.9"),
        mpatches.Patch(color="#f39c12", label="JSS 0.65–0.9"),
        mpatches.Patch(color="#e74c3c", label="JSS < 0.65"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig1_coherence_jss_bar.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2 — 9×4 JSS heatmap
# ---------------------------------------------------------------------------

def fig2_heatmap():
    tasks = ["coherence", "factuality", "preference", "relevance"]

    # Load coherence / preference / relevance JSS
    task_jss = {}
    for task in ("coherence", "preference", "relevance"):
        d = load_task_jss(task)
        task_jss[task] = {m: v[0] for m, v in d.items()}

    # Factuality: use JSS_original (pre-correction)
    fact_df = pd.read_csv(OUTPUT_DIR / "factuality_jss_fixed.csv")
    task_jss["factuality"] = dict(zip(fact_df["model"], fact_df["JSS_original"]))

    # Build matrix: rows = models (same order as MODELS), cols = tasks
    matrix = np.full((len(MODELS), len(tasks)), np.nan)
    for j, task in enumerate(tasks):
        for i, model in enumerate(MODELS):
            v = task_jss[task].get(model)
            if v is not None:
                matrix[i, j] = v

    col_labels = ["Coherence", "Factuality", "Preference", "Relevance"]
    row_labels  = [DISPLAY[m] for m in MODELS]

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")

    # Axes ticks
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(MODELS)))
    ax.set_yticklabels(row_labels)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("JSS", fontsize=10)

    # Cell annotations + hatching
    pref_col = tasks.index("preference")
    relv_col = tasks.index("relevance")
    cohe_col = tasks.index("coherence")

    for i in range(len(MODELS)):
        for j in range(len(tasks)):
            val = matrix[i, j]
            label = f"{val:.2f}" if not np.isnan(val) else "N/A"

            # Dagger on deepseek coherence
            if MODELS[i] == "deepseek" and j == cohe_col:
                label += "\u2020"

            # Text color
            text_color = "black" if (np.isnan(val) or val > 0.4) else "white"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=9, color=text_color)

            # Grey hatching on preference and relevance columns
            if j in (pref_col, relv_col):
                rect = plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, hatch="///", edgecolor="grey",
                    linewidth=0, zorder=3,
                )
                ax.add_patch(rect)

    # Thicker border around coherence column
    rect_cohe = plt.Rectangle(
        (cohe_col - 0.5, -0.5), 1, len(MODELS),
        fill=False, edgecolor="black", linewidth=2.5, zorder=4,
    )
    ax.add_patch(rect_cohe)

    ax.set_title("Judge Sensitivity Score by Model and Task")

    # Footnote
    fig.text(
        0.01, -0.02,
        "Factuality JSS values are pre-T4-correction (raw). "
        "\u2020 DeepSeek coherence has partial data.",
        fontsize=8, ha="left", va="top", style="italic",
    )

    plt.tight_layout()
    out = OUTPUT_DIR / "fig2_jss_heatmap.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 4 — Raw vs corrected factuality JSS grouped bars
# ---------------------------------------------------------------------------

def fig4_raw_vs_corrected():
    df = pd.read_csv(OUTPUT_DIR / "factuality_jss_fixed.csv")
    # Keep original model order
    df = df.set_index("model").reindex(MODELS).reset_index()

    x = np.arange(len(MODELS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    bars_raw  = ax.bar(x - width / 2, df["JSS_original"], width,
                       color="#e74c3c", label="Raw JSS (pre-correction)")
    bars_corr = ax.bar(x + width / 2, df["JSS_fixed"],    width,
                       color="#2ecc71", label="Corrected JSS (post-correction)")

    # Delta annotations above each group
    for i, row in df.iterrows():
        delta = row["JSS_fixed"] - row["JSS_original"]
        group_center = x[i]
        top = max(row["JSS_original"], row["JSS_fixed"]) + 0.02
        ax.text(group_center, top, f"+{delta:.2f}",
                ha="center", va="bottom", fontsize=8, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[m] for m in MODELS], rotation=20, ha="right")
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("JSS")
    ax.set_title("Factuality JSS: Effect of T4 Polarity Correction")
    ax.legend(fontsize=9)
    ax.axhline(y=1.0, color="grey", linestyle=":", linewidth=0.8)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig4_factuality_raw_vs_corrected.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("TASK 5 — Publication Figures")
    print("=" * 70)

    created = []

    print("\nGenerating Figure 1 (coherence bar chart)...")
    out1 = fig1_coherence_bar()
    created.append(out1)

    print("Generating Figure 2 (JSS heatmap)...")
    out2 = fig2_heatmap()
    created.append(out2)

    print("Figure 3 skipped (cluttered)")

    print("Generating Figure 4 (raw vs corrected factuality JSS)...")
    out4 = fig4_raw_vs_corrected()
    created.append(out4)

    print("\nFigures created:")
    for p in created:
        print(f"  {p}")
    print("fig3 skipped (cluttered)")
    print("\nTASK 5 COMPLETE")


if __name__ == "__main__":
    main()
