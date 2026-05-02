import csv
import os

MODEL_NAME_MAP = {
    "claude-haiku": "claude-haiku-4-5",
    "claude-sonnet": "claude-sonnet-4-5",
    "deepseek": "deepseek-r1",
    "gemini-flash": "gemini-2.5-flash",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o": "gpt-4o",
    "llama3-70b": "llama-3.1-70b",
    "mistral-7b": "mistral-7b",
    "qwen": "qwen-2.5-72b",
}

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "outputs", "factuality_jss_fixed.csv")

rows = []
with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        model_key = row["model"]
        paper_name = MODEL_NAME_MAP.get(model_key, model_key)
        jss_raw = round(float(row["JSS_original"]), 3)
        jss_corrected = round(float(row["JSS_fixed"]), 3)
        delta = round(jss_corrected - jss_raw, 3)
        rows.append((paper_name, jss_raw, jss_corrected, delta))

rows.sort(key=lambda r: (-r[2], -r[1]))

print("\nConsole table (sorted by corrected JSS desc):")
print(f"{'Model':<22} {'JSS raw':>9} {'JSS corr':>10} {'Delta':>7}")
print("-" * 52)
for name, raw, corr, delta in rows:
    print(f"{name:<22} {raw:>9.3f} {corr:>10.3f} {delta:>+7.3f}")

print("\nLaTeX midrule rows:")
for name, raw, corr, delta in rows:
    print(f"{name:<22} & {raw:.3f} & {corr:.3f} & {delta:+.3f} \\\\")
