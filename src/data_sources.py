"""
JudgeSense v2 data sources — real loaders with a mandatory provenance chain.

Every item returned by this module is drawn from a specific record of a
specific split of a public dataset on the Hugging Face Hub, and carries a
`source` record identifying it (dataset id, config, split, record id).

Failure policy (load-bearing, see docs/V2_ARCHITECTURE.md §1.3):
if a source cannot be loaded — network policy, missing credentials, schema
drift, anything — the loader raises DataSourceUnavailableError. There is no
fallback to synthetic, cached-in-code, or placeholder items, and none may
ever be added. The v1 dataset crisis was caused by hardcoded items being
labeled as benchmark-sourced; a loud failure is the correct behavior when
data is unavailable.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

LOADER_VERSION = "2.0.0"

# Number of unique items each task loader targets (audit-gated minimum is
# declared separately in data/audit_config.json).
DEFAULT_ITEMS_PER_TASK = 250


class DataSourceUnavailableError(RuntimeError):
    """A real data source could not be loaded. Never catch-and-fallback."""

    def __init__(self, dataset_id: str, cause: Exception | str):
        self.dataset_id = dataset_id
        self.cause = cause
        super().__init__(
            f"Could not load required source dataset '{dataset_id}': {cause}\n"
            "JudgeSense v2 refuses to substitute synthetic or placeholder "
            "data. Fix connectivity/credentials and re-run; do not bypass "
            "this error."
        )


class DataSourceSchemaError(RuntimeError):
    """A source loaded but its schema does not match what we validated against."""


@dataclass
class SourceRecord:
    """Provenance chain for a single item. All fields are mandatory."""

    source_dataset: str
    source_config: Optional[str]
    source_split: str
    source_record_id: str
    source_fields: Dict[str, str]
    retrieved_at: str
    loader_version: str = LOADER_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SourceItem:
    """A single unit of judgment material with real provenance."""

    item_id: str
    task_type: str
    text: str                       # the text (or candidate pair) being judged
    ground_truth_label: str
    source: SourceRecord
    extra: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["source"] = self.source.to_dict()
        return d


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_hf_dataset(dataset_id: str, config: Optional[str], split: str):
    """Load a Hugging Face dataset split; any failure is loud and typed."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise DataSourceUnavailableError(
            dataset_id, f"the 'datasets' package is not installed ({exc})"
        ) from exc
    try:
        if config is not None:
            return load_dataset(dataset_id, config, split=split)
        return load_dataset(dataset_id, split=split)
    except Exception as exc:
        raise DataSourceUnavailableError(dataset_id, exc) from exc


def _require_columns(ds, dataset_id: str, columns: List[str]) -> None:
    missing = [c for c in columns if c not in ds.column_names]
    if missing:
        raise DataSourceSchemaError(
            f"Dataset '{dataset_id}' loaded but is missing expected columns "
            f"{missing}; refusing to guess. Found: {ds.column_names}"
        )


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# ── Task 1: factuality ← TruthfulQA ─────────────────────────────────────────

def load_factuality_items(
    n_items: int = DEFAULT_ITEMS_PER_TASK,
    seed: int = 42,
    dataset_id: str = "truthful_qa",
    config: str = "generation",
    split: str = "validation",
    _loader: Optional[Callable] = None,
) -> List[SourceItem]:
    """
    Factuality items from TruthfulQA (generation config).

    Each source question contributes up to two items: its `best_answer`
    (label: accurate) and its first `incorrect_answer` (label: inaccurate),
    each phrased as a "Q: ... A: ..." statement so the judged text is
    self-contained. Labels come from TruthfulQA's own annotation, not from
    this repository.
    """
    ds = (_loader or _load_hf_dataset)(dataset_id, config, split)
    _require_columns(ds, dataset_id, ["question", "best_answer", "incorrect_answers"])

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    retrieved_at = _now_iso()
    items: List[SourceItem] = []
    for row_idx in indices:
        if len(items) >= n_items:
            break
        row = ds[row_idx]
        question = (row["question"] or "").strip()
        best = (row["best_answer"] or "").strip()
        incorrect = [a.strip() for a in (row["incorrect_answers"] or []) if a and a.strip()]
        if not question or not best:
            continue

        def _mk(answer: str, label: str, which: str) -> SourceItem:
            text = f"Q: {question}\nA: {answer}"
            return SourceItem(
                item_id=f"fact_tqa_{row_idx}_{which}",
                task_type="factuality",
                text=text,
                ground_truth_label=label,
                source=SourceRecord(
                    source_dataset=dataset_id,
                    source_config=config,
                    source_split=split,
                    source_record_id=f"{split}[{row_idx}]",
                    source_fields={"question": question, "answer_field": which},
                    retrieved_at=retrieved_at,
                ),
            )

        items.append(_mk(best, "accurate", "best_answer"))
        if len(items) < n_items and incorrect:
            items.append(_mk(incorrect[0], "inaccurate", "incorrect_answers[0]"))

    if len(items) < n_items:
        raise DataSourceSchemaError(
            f"TruthfulQA yielded only {len(items)} usable items "
            f"(requested {n_items}); refusing to pad."
        )
    return items


# ── Task 2: coherence ← SummEval ────────────────────────────────────────────

def load_coherence_items(
    n_items: int = DEFAULT_ITEMS_PER_TASK,
    seed: int = 42,
    dataset_id: str = "mteb/summeval",
    config: Optional[str] = None,
    split: str = "test",
    _loader: Optional[Callable] = None,
) -> List[SourceItem]:
    """
    Coherence items from SummEval: (source document, machine summary) pairs
    with expert coherence annotations (1-5). The ground truth label is the
    mean expert coherence rating rounded to the nearest integer; the raw
    float is preserved in `extra["coherence_raw"]`.
    """
    ds = (_loader or _load_hf_dataset)(dataset_id, config, split)
    _require_columns(ds, dataset_id, ["text", "machine_summaries", "coherence"])

    rng = random.Random(seed)
    candidates = []
    for row_idx in range(len(ds)):
        row = ds[row_idx]
        summaries = row["machine_summaries"] or []
        scores = row["coherence"] or []
        if len(summaries) != len(scores):
            raise DataSourceSchemaError(
                f"SummEval row {row_idx}: {len(summaries)} summaries vs "
                f"{len(scores)} coherence scores; refusing to align by guess."
            )
        for sum_idx, (summary, score) in enumerate(zip(summaries, scores)):
            if summary and summary.strip():
                candidates.append((row_idx, sum_idx, summary.strip(), float(score)))

    # Distinct SummEval system summaries are sometimes byte-identical while
    # carrying different expert coherence ratings. Left alone, that ships the
    # same text twice under different item_ids — once with contradictory ground
    # truth (observed: labels 4 and 5 for one identical summary), which no judge
    # can satisfy on both. Group by exact text and keep one representative;
    # where the rounded expert labels disagree there is no defensible single
    # ground truth, so the text is dropped rather than arbitrated.
    by_text: Dict[str, List[tuple]] = {}
    for cand in candidates:
        by_text.setdefault(cand[2], []).append(cand)

    deduped: List[tuple] = []
    n_dropped_conflict = 0
    n_merged = 0
    for text, group in by_text.items():
        if len(group) == 1:
            deduped.append(group[0])
            continue
        labels = {int(min(5, max(1, round(c[3])))) for c in group}
        if len(labels) > 1:
            n_dropped_conflict += 1
            continue
        n_merged += 1
        deduped.append(sorted(group, key=lambda c: (c[0], c[1]))[0])

    if n_dropped_conflict or n_merged:
        print(
            f"  [coherence] deduplicated identical summaries: {n_merged} merged, "
            f"{n_dropped_conflict} dropped for conflicting expert labels"
        )
    candidates = deduped
    rng.shuffle(candidates)

    if len(candidates) < n_items:
        raise DataSourceSchemaError(
            f"SummEval yielded only {len(candidates)} usable (doc, summary) "
            f"pairs (requested {n_items}); refusing to pad."
        )

    retrieved_at = _now_iso()
    items: List[SourceItem] = []
    for row_idx, sum_idx, summary, score in candidates[:n_items]:
        rounded = int(min(5, max(1, round(score))))
        items.append(
            SourceItem(
                item_id=f"cohe_summeval_{row_idx}_{sum_idx}",
                task_type="coherence",
                text=summary,
                ground_truth_label=str(rounded),
                source=SourceRecord(
                    source_dataset=dataset_id,
                    source_config=config,
                    source_split=split,
                    source_record_id=f"{split}[{row_idx}].machine_summaries[{sum_idx}]",
                    source_fields={"annotation_field": "coherence"},
                    retrieved_at=retrieved_at,
                ),
                extra={"coherence_raw": score},
            )
        )
    return items


# ── Task 3: relevance ← BEIR (SciFact) ──────────────────────────────────────

def load_relevance_items(
    n_items: int = DEFAULT_ITEMS_PER_TASK,
    seed: int = 42,
    corpus_id: str = "BeIR/scifact",
    qrels_id: str = "BeIR/scifact-qrels",
    _loader: Optional[Callable] = None,
) -> List[SourceItem]:
    """
    Relevance items from BEIR SciFact: a query, its qrels-relevant document,
    and a deterministically sampled non-relevant document from the same
    corpus. Both documents are real corpus records; the pairing procedure
    (not the texts or the relevance label) is constructed, and both document
    ids are recorded in the provenance chain.
    """
    loader = _loader or _load_hf_dataset
    corpus = loader(corpus_id, "corpus", "corpus")
    queries = loader(corpus_id, "queries", "queries")
    qrels = loader(qrels_id, None, "train")
    _require_columns(corpus, corpus_id, ["_id", "text"])
    _require_columns(queries, corpus_id, ["_id", "text"])
    _require_columns(qrels, qrels_id, ["query-id", "corpus-id", "score"])

    corpus_by_id = {str(row["_id"]): (row["text"] or "").strip() for row in corpus}
    query_by_id = {str(row["_id"]): (row["text"] or "").strip() for row in queries}
    all_doc_ids = sorted(corpus_by_id.keys())

    relevant: Dict[str, str] = {}
    relevant_sets: Dict[str, set] = {}
    for row in qrels:
        qid, did, score = str(row["query-id"]), str(row["corpus-id"]), int(row["score"])
        if score > 0:
            relevant.setdefault(qid, did)
            relevant_sets.setdefault(qid, set()).add(did)

    rng = random.Random(seed)
    qids = sorted(relevant.keys())
    rng.shuffle(qids)

    retrieved_at = _now_iso()
    items: List[SourceItem] = []
    for qid in qids:
        if len(items) >= n_items:
            break
        query = query_by_id.get(qid, "")
        pos_id = relevant[qid]
        pos_text = corpus_by_id.get(pos_id, "")
        if not query or not pos_text:
            continue
        # Deterministic negative: sample until we hit a doc not relevant to qid.
        neg_id = None
        for _ in range(100):
            cand = all_doc_ids[rng.randrange(len(all_doc_ids))]
            if cand not in relevant_sets[qid] and corpus_by_id[cand]:
                neg_id = cand
                break
        if neg_id is None:
            continue
        items.append(
            SourceItem(
                item_id=f"relv_scifact_{qid}",
                task_type="relevance",
                text=query,
                # Must name one of the `extra` candidate keys below: the builder
                # resolves the ground truth to a display position by matching this
                # value against candidate_map. "relevant_candidate" transposes the
                # words and matches nothing.
                ground_truth_label="candidate_relevant",
                source=SourceRecord(
                    source_dataset=corpus_id,
                    source_config="corpus+queries",
                    source_split="corpus/queries/qrels-train",
                    source_record_id=f"query[{qid}]",
                    source_fields={
                        "relevant_doc_id": pos_id,
                        "nonrelevant_doc_id": neg_id,
                        "qrels_dataset": qrels_id,
                        "pairing": "constructed: qrels-positive vs seeded corpus sample",
                    },
                    retrieved_at=retrieved_at,
                ),
                extra={
                    "candidate_relevant": corpus_by_id[pos_id],
                    "candidate_nonrelevant": corpus_by_id[neg_id],
                },
            )
        )

    if len(items) < n_items:
        raise DataSourceSchemaError(
            f"BEIR SciFact yielded only {len(items)} usable query-document "
            f"pairs (requested {n_items}); refusing to pad."
        )
    return items


# ── Task 4: preference ← MT-Bench human judgments ───────────────────────────

def load_preference_items(
    n_items: int = DEFAULT_ITEMS_PER_TASK,
    seed: int = 42,
    dataset_id: str = "lmsys/mt_bench_human_judgments",
    config: Optional[str] = None,
    split: str = "human",
    _loader: Optional[Callable] = None,
) -> List[SourceItem]:
    """
    Preference items from MT-Bench human judgments: real response pairs
    (model_a vs model_b, turn 1) with real human preference votes. Where a
    (question, model_a, model_b) pair has multiple votes, the majority is
    used and the tally recorded; ties and no-majority pairs are excluded.
    """
    ds = (_loader or _load_hf_dataset)(dataset_id, config, split)
    _require_columns(
        ds, dataset_id,
        ["question_id", "model_a", "model_b", "winner", "conversation_a", "conversation_b", "turn"],
    )

    def _first_assistant(conv) -> str:
        for msg in conv or []:
            if msg.get("role") == "assistant":
                return (msg.get("content") or "").strip()
        return ""

    votes: Dict[tuple, Dict[str, int]] = {}
    texts: Dict[tuple, tuple] = {}
    for row_idx in range(len(ds)):
        row = ds[row_idx]
        if int(row["turn"]) != 1:
            continue
        qid = str(row["question_id"])
        model_a, model_b = str(row["model_a"]), str(row["model_b"])

        # MT-Bench contains BOTH (model_a=X, model_b=Y) and (X and Y swapped)
        # rows for the same question. Keyed by ordered pair those become two
        # separate items encoding the SAME comparison — and because the winner
        # is recorded positionally, the two items carry OPPOSITE ground truth
        # for byte-identical candidate text. Canonicalize to an unordered model
        # pair, flipping the winner and the response texts for rows that arrive
        # in non-canonical order, so each comparison is one item with its votes
        # pooled across both presentation orders.
        flipped = model_a > model_b
        first, second = (model_b, model_a) if flipped else (model_a, model_b)
        key = (qid, first, second)

        winner = str(row["winner"])
        if flipped:
            winner = {"model_a": "model_b", "model_b": "model_a"}.get(winner, winner)

        votes.setdefault(key, {})
        votes[key][winner] = votes[key].get(winner, 0) + 1
        if key not in texts:
            ra = _first_assistant(row["conversation_a"])
            rb = _first_assistant(row["conversation_b"])
            if flipped:
                ra, rb = rb, ra
            question = ""
            for msg in row["conversation_a"] or []:
                if msg.get("role") == "user":
                    question = (msg.get("content") or "").strip()
                    break
            texts[key] = (question, ra, rb, row_idx)

    rng = random.Random(seed)
    keys = sorted(votes.keys())
    rng.shuffle(keys)

    retrieved_at = _now_iso()
    items: List[SourceItem] = []
    n_excluded_tie = 0
    n_excluded_duplicate = 0
    seen_content: set = set()
    for key in keys:
        if len(items) >= n_items:
            break
        tally = votes[key]
        contested = {k: v for k, v in tally.items() if k in ("model_a", "model_b")}
        if not contested:
            n_excluded_tie += 1
            continue
        best = max(contested.values())
        winners = [k for k, v in contested.items() if v == best]
        if len(winners) != 1:
            n_excluded_tie += 1
            continue
        question, ra, rb, row_idx = texts[key]
        if not question or not ra or not rb:
            continue

        # Two different model pairs can produce byte-identical responses for the
        # same question (observed on mt_bench question 135, where both pairs
        # returned the same structured answer). The judge sees content, not
        # model names, so those are one comparison and shipping both would
        # inflate the item count with a duplicate.
        content_signature = (question, ra, rb)
        if content_signature in seen_content:
            n_excluded_duplicate += 1
            continue
        seen_content.add(content_signature)

        label = "candidate_1" if winners[0] == "model_a" else "candidate_2"
        qid, model_a, model_b = key
        items.append(
            SourceItem(
                item_id=f"pref_mtbench_{qid}_{_stable_hash(model_a + '|' + model_b)}",
                task_type="preference",
                text=question,
                ground_truth_label=label,
                source=SourceRecord(
                    source_dataset=dataset_id,
                    source_config=config,
                    source_split=split,
                    source_record_id=f"question_id={qid};model_a={model_a};model_b={model_b};turn=1",
                    source_fields={
                        "first_seen_row": str(row_idx),
                        "vote_tally": str(tally),
                        "label_rule": "majority of human votes; ties excluded",
                    },
                    retrieved_at=retrieved_at,
                ),
                extra={"candidate_1": ra, "candidate_2": rb},
            )
        )

    if len(items) < n_items:
        raise DataSourceSchemaError(
            f"MT-Bench human judgments yielded only {len(items)} usable "
            f"majority-labeled pairs (requested {n_items}, "
            f"{n_excluded_tie} excluded as tie/no-majority, "
            f"{n_excluded_duplicate} as duplicate content); refusing to pad."
        )
    return items


TASK_LOADERS = {
    "factuality": load_factuality_items,
    "coherence": load_coherence_items,
    "relevance": load_relevance_items,
    "preference": load_preference_items,
}
