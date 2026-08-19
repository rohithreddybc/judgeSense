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
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

LOADER_VERSION = "2.1.0"

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


DIFFICULTIES = ("hard", "easy")


def _require_difficulty(difficulty: str) -> None:
    if difficulty not in DIFFICULTIES:
        raise ValueError(
            f"difficulty must be one of {DIFFICULTIES}, got {difficulty!r}"
        )


# ── Lexical retrieval (hard-negative mining for the relevance task) ─────────
#
# A compact BM25 index over the real corpus, used ONLY to *select* which real
# corpus document serves as the non-relevant candidate. It never generates
# text and never assigns relevance labels: the relevance ground truth remains
# qrels, exactly as before. Scores are recorded per item so the difficulty of
# every pairing is auditable.

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class _BM25Index:
    """Minimal Okapi BM25 over an in-memory corpus (no extra dependencies)."""

    def __init__(self, docs: Dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = sorted(docs.keys())
        self.doc_len: List[int] = []
        self.postings: Dict[str, Dict[int, int]] = defaultdict(dict)
        for i, did in enumerate(self.doc_ids):
            toks = _tokenize(docs[did])
            self.doc_len.append(len(toks))
            for term, tf in Counter(toks).items():
                self.postings[term][i] = tf
        self.n_docs = len(self.doc_ids)
        self.avg_len = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0

    def scores(self, query: str) -> Dict[int, float]:
        """BM25 score for every doc sharing at least one term with `query`."""
        acc: Dict[int, float] = defaultdict(float)
        for term in set(_tokenize(query)):
            postings = self.postings.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for i, tf in postings.items():
                norm = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len)
                acc[i] += idf * tf * (self.k1 + 1) / norm
        return acc

    def ranked(self, query: str) -> List[Tuple[str, float]]:
        """All docs ranked by descending BM25 score, doc-id tiebreak.

        Docs sharing no term with the query score 0.0 and rank after every
        scored doc; the doc-id tiebreak keeps the ordering deterministic.
        """
        acc = self.scores(query)
        order = sorted(
            range(self.n_docs),
            key=lambda i: (-acc.get(i, 0.0), self.doc_ids[i]),
        )
        return [(self.doc_ids[i], acc.get(i, 0.0)) for i in order]


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
    corpus_id: str = "BeIR/trec-covid",
    qrels_id: str = "BeIR/trec-covid-qrels",
    qrels_split: str = "test",
    difficulty: str = "hard",
    _loader: Optional[Callable] = None,
) -> List[SourceItem]:
    """
    Relevance items from BEIR TREC-COVID: a topic, a document a human assessor
    judged FULLY RELEVANT (graded score 2), and a document the SAME assessment
    process judged EXPLICITLY NON-RELEVANT (score 0) for that topic.

    This is the substantive difference from a positives-only collection such as
    SciFact. There, the negative could only be a document ABSENT from an
    incomplete qrels, so a topically similar distractor might in fact be
    relevant-but-unjudged, and the item would penalise a defensible answer.
    TREC-COVID carries graded human judgements including 41k+ explicit
    non-relevant assessments, so here BOTH candidates carry a real human
    relevance label and the negative is one a human affirmatively rejected.
    Nothing in this module assigns or infers relevance: the labels are TREC's.

    difficulty selects WHICH human-judged non-relevant document is shown —
    never its label:

    - "hard" (default): among the topic's explicit non-relevant documents, the
      one with the HIGHEST BM25 score against the topic — a same-neighbourhood
      distractor a human still rejected. A random non-relevant document is so
      unrelated that the task cannot discriminate between judges.
    - "easy": a seeded random draw from the same explicit-non-relevant pool.

    Each item records both documents' human relevance grade and the negative's
    BM25 score and within-pool rank, so difficulty is auditable per item.
    """
    _require_difficulty(difficulty)
    loader = _loader or _load_hf_dataset
    corpus = loader(corpus_id, "corpus", "corpus")
    queries = loader(corpus_id, "queries", "queries")
    qrels = loader(qrels_id, None, qrels_split)
    _require_columns(corpus, corpus_id, ["_id", "text"])
    _require_columns(queries, corpus_id, ["_id", "text"])
    _require_columns(qrels, qrels_id, ["query-id", "corpus-id", "score"])

    query_by_id = {str(row["_id"]): (row["text"] or "").strip() for row in queries}

    # Graded human judgements per topic: fully-relevant positives (score 2) and
    # explicitly-rejected negatives (score 0). Partial-relevance (score 1) is
    # used for NEITHER candidate — it is neither a clean positive nor a clean
    # negative — so every shipped candidate is an unambiguous human judgement.
    positives: Dict[str, List[str]] = defaultdict(list)
    negatives: Dict[str, List[str]] = defaultdict(list)
    judged_ids: set = set()
    for row in qrels:
        qid, did, score = str(row["query-id"]), str(row["corpus-id"]), int(row["score"])
        if score >= 2:
            positives[qid].append(did); judged_ids.add(did)
        elif score == 0:
            negatives[qid].append(did); judged_ids.add(did)

    # Only the judged documents' text is needed; the TREC-COVID corpus is large,
    # so keep just those rather than the whole collection in memory.
    corpus_by_id: Dict[str, str] = {}
    for row in corpus:
        did = str(row["_id"])
        if did in judged_ids:
            corpus_by_id[did] = (row["text"] or "").strip()

    index = _BM25Index({d: corpus_by_id[d] for d in judged_ids if corpus_by_id.get(d)})

    rng = random.Random(seed)
    qids = sorted(q for q in positives if negatives.get(q))
    rng.shuffle(qids)
    if not qids:
        raise DataSourceSchemaError(
            f"{corpus_id}: no topic has both a fully-relevant (score 2) and an "
            "explicit non-relevant (score 0) human judgement."
        )

    # Distribute items across the available topics. TREC-COVID has ~50 topics
    # but many graded judgements each, so several distinct (relevant,
    # non-relevant) document pairs are drawn per topic; each is a separate
    # judgement over real human-labelled documents.
    per_query_cap = max(1, math.ceil(n_items / len(qids)))

    def _texts_ok(a: str, b: str) -> bool:
        return bool(a) and bool(b) and a != b

    retrieved_at = _now_iso()
    items: List[SourceItem] = []
    used_pairs: set = set()
    n_excluded_duplicate = 0

    # Round over topics so items are spread across topics rather than exhausting
    # one before the next; within a topic, hardest negatives first.
    ranked_neg: Dict[str, List[Tuple[str, float]]] = {}
    for qid in qids:
        scores = index.scores(query_by_id.get(qid, ""))
        idx_of = {index.doc_ids[i]: i for i in range(index.n_docs)}
        scored = [
            (did, scores.get(idx_of[did], 0.0))
            for did in negatives[qid]
            if did in idx_of and corpus_by_id.get(did)
        ]
        if difficulty == "hard":
            scored.sort(key=lambda t: (-t[1], t[0]))
        else:
            rng.shuffle(scored)
        ranked_neg[qid] = scored

    emitted_per_query: Dict[str, int] = defaultdict(int)
    progress = True
    while len(items) < n_items and progress:
        progress = False
        for qid in qids:
            if len(items) >= n_items:
                break
            if emitted_per_query[qid] >= per_query_cap:
                continue
            j = emitted_per_query[qid]
            pos_ids = [d for d in positives[qid] if corpus_by_id.get(d)]
            negs = ranked_neg[qid]
            if j >= len(negs) or not pos_ids:
                continue
            pos_id = pos_ids[j % len(pos_ids)]
            neg_id, neg_score = negs[j]
            pos_text, neg_text = corpus_by_id.get(pos_id, ""), corpus_by_id.get(neg_id, "")
            emitted_per_query[qid] += 1
            progress = True
            if not _texts_ok(pos_text, neg_text):
                continue
            pair_sig = (min(pos_text, neg_text), max(pos_text, neg_text))
            if pair_sig in used_pairs:
                n_excluded_duplicate += 1
                continue
            used_pairs.add(pair_sig)
            items.append(
                SourceItem(
                    item_id=f"relv_treccovid_{qid}_{j}",
                    task_type="relevance",
                    text=query_by_id[qid],
                    ground_truth_label="candidate_relevant",
                    source=SourceRecord(
                        source_dataset=corpus_id,
                        source_config="corpus+queries",
                        source_split=f"corpus/queries/qrels-{qrels_split}",
                        source_record_id=f"query[{qid}]#pair{j}",
                        source_fields={
                            "relevant_doc_id": pos_id,
                            "relevant_human_grade": "2 (fully relevant)",
                            "nonrelevant_doc_id": neg_id,
                            "nonrelevant_human_grade": "0 (explicitly non-relevant)",
                            "qrels_dataset": qrels_id,
                            "pairing": (
                                "both candidates human-judged: relevant=TREC grade 2, "
                                "non-relevant=TREC grade 0; distractor selected by BM25 "
                                "hardness within the explicit-non-relevant pool"
                            ),
                            "difficulty": difficulty,
                            "neg_bm25_score": f"{neg_score:.4f}",
                            "neg_bm25_rank_in_pool": str(j + 1),
                            "n_explicit_negatives": str(len(negs)),
                        },
                        retrieved_at=retrieved_at,
                    ),
                    extra={
                        "candidate_relevant": pos_text,
                        "candidate_nonrelevant": neg_text,
                    },
                )
            )

    if len(items) < n_items:
        raise DataSourceSchemaError(
            f"BEIR TREC-COVID yielded only {len(items)} usable (relevant, "
            f"explicit-non-relevant) pairs over {len(qids)} topics (requested "
            f"{n_items}, {n_excluded_duplicate} excluded as duplicate document "
            "pairs); refusing to pad."
        )
    return items


# ── Task 4: preference ← MT-Bench human judgments ───────────────────────────

def load_preference_items(
    n_items: int = DEFAULT_ITEMS_PER_TASK,
    seed: int = 42,
    dataset_id: str = "lmsys/mt_bench_human_judgments",
    config: Optional[str] = None,
    split: str = "human",
    difficulty: str = "hard",
    min_votes: int = 2,
    _loader: Optional[Callable] = None,
) -> List[SourceItem]:
    """
    Preference items from MT-Bench human judgments: real response pairs
    (model_a vs model_b, turn 1) with real human preference votes. Where a
    (question, model_a, model_b) pair has multiple votes, the majority is
    used and the tally recorded; ties and no-majority pairs are excluded.

    ``min_votes`` (default 2) requires each shipped label to rest on at least
    that many human votes. At the default, no preference item's ground truth
    is a single annotator's opinion: every label is a majority of two or more
    independent human votes. The source contains 324 turn-1 comparisons with a
    strict multi-vote majority, so the 250-item benchmark is drawn entirely
    from multiply-annotated comparisons. Setting ``min_votes=1`` restores the
    prior behaviour (single-vote labels admitted) and is retained only for
    reproducing the earlier release.

    difficulty selects WHICH majority-labeled comparisons ship — never the
    label, which is always the recorded human majority:

    - "hard" (default): comparisons ordered by ascending vote-margin ratio
      (winner votes − loser votes, over all votes incl. ties), then ascending
      absolute margin — i.e. the most CONTESTED comparisons first. The
      previous uniform selection was dominated by unanimous pairs, and 11/13
      judges scored exactly 1.000.
    - "easy": the same pool ordered by descending margin ratio then
      descending margin — the most decisively won comparisons first.

    Each item records its vote margin, margin ratio, and total vote count in
    `source_fields`, so per-item difficulty is auditable.
    """
    _require_difficulty(difficulty)
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

    def _margin(key) -> Optional[Tuple[int, int, float]]:
        """(margin, total votes, margin ratio) for a majority-labeled pair.

        margin = winner votes − runner-up votes over the directional votes;
        total counts every vote including ties, so a 2-1-with-2-ties pair
        ranks as more contested than a bare 2-1. Returns None where no strict
        majority exists, or where the total vote count is below ``min_votes``
        (excluded regardless of difficulty): a label resting on fewer than
        ``min_votes`` human votes is not a multi-annotator judgement.
        """
        tally = votes[key]
        total = sum(tally.values())
        if total < min_votes:
            return None
        directional = {k: v for k, v in tally.items() if k in ("model_a", "model_b")}
        if not directional:
            return None
        best = max(directional.values())
        if len([k for k, v in directional.items() if v == best]) != 1:
            return None
        margin = best - (sum(directional.values()) - best)
        return margin, total, margin / total

    # Stable sort after the seeded shuffle: difficulty orders the pool,
    # the shuffle breaks ranking ties deterministically per seed.
    margins = {key: _margin(key) for key in keys}
    if difficulty == "hard":
        keys.sort(key=lambda k: (margins[k][2], margins[k][0]) if margins[k] else (2.0, 0))
    else:
        keys.sort(key=lambda k: (-margins[k][2], -margins[k][0]) if margins[k] else (2.0, 0))

    retrieved_at = _now_iso()
    items: List[SourceItem] = []
    n_excluded_tie = 0
    n_excluded_duplicate = 0
    seen_content: set = set()
    for key in keys:
        if len(items) >= n_items:
            break
        tally = votes[key]
        if margins[key] is None:
            n_excluded_tie += 1
            continue
        margin, total_votes, margin_ratio = margins[key]
        contested = {k: v for k, v in tally.items() if k in ("model_a", "model_b")}
        best = max(contested.values())
        winners = [k for k, v in contested.items() if v == best]
        question, ra, rb, row_idx = texts[key]
        if not question or not ra or not rb or ra == rb:
            continue

        # Two different model pairs can produce byte-identical responses for the
        # same question (observed on mt_bench question 135, where both pairs
        # returned the same structured answer). The judge sees candidate
        # content — not model names and not the question — so the dedup key is
        # the unordered response pair: an exact repeat is a duplicate item, and
        # a role-reversed repeat would be an outright label contradiction
        # (the A/B swap design displays both orders of every pair).
        content_signature = (min(ra, rb), max(ra, rb))
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
                        "label_rule": f"majority of >= {min_votes} human votes; ties and single-vote items excluded",
                        "difficulty": difficulty,
                        "vote_margin": str(margin),
                        "vote_margin_ratio": f"{margin_ratio:.4f}",
                        "total_votes": str(total_votes),
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
