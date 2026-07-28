"""
Generate Croissant metadata for the v2 dataset FROM the built data.

Metadata is derived from the files on disk, never hand-edited: record
counts, unique-item counts, label spaces, and source-dataset lists are all
computed. The generator refuses to run when the built dataset is absent —
metadata without data is how the v1 croissant ended up describing items
that were never loaded from the benchmarks it names.

`rai:hasSyntheticData` is set accurately for the v2 design: TRUE, with an
explanation — the judged texts include machine-generated content from the
source benchmarks themselves (SummEval machine summaries, MT-Bench model
responses), and judge prompts are author-written templates; item texts and
ground-truth labels are drawn verbatim from the source datasets. (Note: the
v1 file judgesense-benchmark/judgesense_croissant.json declares
hasSyntheticData: false, which is inaccurate for v1's hand-authored items;
that file describes the previously published artifact and is left
untouched.)

Usage:
    python scripts/generate_croissant_v2.py --data-dir data/v2 \
        --output data/v2/croissant_v2.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

TASKS = ("factuality", "coherence", "relevance", "preference")


class DatasetNotBuiltError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_croissant(data_dir: Path) -> dict:
    data_dir = Path(data_dir)
    missing = [t for t in TASKS if not (data_dir / f"{t}.jsonl").exists()]
    if missing:
        raise DatasetNotBuiltError(
            f"v2 dataset not built: missing {missing} under {data_dir}. "
            "Build it with src/dataset_builder_v2.py (requires access to the "
            "real source datasets); metadata is only ever generated from "
            "built data."
        )

    distribution = []
    record_sets = []
    source_datasets = set()
    for task in TASKS:
        path = data_dir / f"{task}.jsonl"
        records = load_jsonl(path)
        if not records:
            raise DatasetNotBuiltError(f"{path} exists but is empty; refusing.")
        labels = sorted({str(r.get("ground_truth_label")) for r in records})
        unique_items = len({r["item_id"] for r in records})
        for rec in records:
            source_datasets.add(rec["source"]["source_dataset"])

        distribution.append({
            "@type": "cr:FileObject",
            "@id": f"{task}-v2-jsonl",
            "name": f"{task}.jsonl",
            "description": (
                f"{len(records)} records over {unique_items} unique source "
                f"items for the {task} task. Every record carries a per-item "
                "provenance chain (source dataset, config, split, record id)."
            ),
            "contentUrl": f"data/{task}.jsonl",
            "encodingFormat": "application/jsonlines",
            "sha256": sha256_file(path),
        })
        record_sets.append({
            "@type": "cr:RecordSet",
            "@id": f"{task}-v2-records",
            "name": f"{task} prompt-paraphrase records (v2)",
            "description": (
                f"{len(records)} records, {unique_items} unique items, "
                f"label space {labels}."
            ),
            "source": {"fileObject": {"@id": f"{task}-v2-jsonl"}},
        })

    return {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
            "rai": "http://mlcommons.org/croissant/RAI/",
            "dct": "http://purl.org/dc/terms/",
            "sc": "https://schema.org/",
            "conformsTo": "dct:conformsTo",
        },
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.1",
        "name": "JudgeSense-v2",
        "description": (
            "JudgeSense v2: prompt-paraphrase pairs for measuring prompt "
            "sensitivity in LLM-as-a-Judge systems. All judged items are "
            "loaded from public source datasets with per-item provenance; "
            "pairwise tasks present both candidate orderings by design."
        ),
        "version": "2.0-rebuild",
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "creator": {"@type": "Person", "name": "Anonymous Author"},
        "isLiveDataset": False,
        "distribution": distribution,
        "recordSet": record_sets,
        "rai:hasSyntheticData": True,
        "rai:syntheticDataExplanation": (
            "The judged texts include machine-generated content drawn "
            "verbatim from the source benchmarks (SummEval machine "
            "summaries; MT-Bench model responses). Judge prompt templates "
            "are author-written paraphrases. Item texts and ground-truth "
            "labels are taken verbatim from the source datasets — none are "
            "authored in this repository. Source datasets present in the "
            f"built data: {sorted(source_datasets)}."
        ),
        "rai:dataCollection": (
            "Items are loaded programmatically from public datasets on the "
            "Hugging Face Hub by src/data_sources.py; every record carries "
            "the source dataset id, config, split, and per-record id. The "
            "build fails if any source is unavailable — no items are "
            "authored or synthesized by the builder."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate v2 Croissant metadata from built data")
    parser.add_argument("--data-dir", default="data/v2")
    parser.add_argument("--output", default="data/v2/croissant_v2.json")
    args = parser.parse_args()

    croissant = build_croissant(Path(args.data_dir))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(croissant, indent=2) + "\n", encoding="utf-8")
    print(f"[written] {out}")


if __name__ == "__main__":
    main()
