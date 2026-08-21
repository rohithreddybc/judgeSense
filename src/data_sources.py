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

    # TruthfulQA uses a canonical refusal ("I have no comment") as the safe
    # answer to unanswerable or ill-posed questions. Judging "is this factually
    # correct?" on a refusal is nonsensical, so those rows are skipped. Questions
    # addressed to the model ("Do you...?", "Are you...?") are indexical rather
    # than factual and are skipped for the same reason.
    _REFUSAL = ("i have no comment", "no comment", "i have no idea",
                "it's not possible", "i'm not sure", "unknown")
    def _ill_posed(question: str, best: str) -> bool:
        b = best.lower().strip().rstrip(".")
        if any(b == r or b.startswith(r) for r in _REFUSAL):
            return True
        ql = question.lower()
        return ql.startswith(("do you", "are you", "have you", "can you", "will you"))

    retrieved_at = _now_iso()
    items: List[SourceItem] = []
    seq = 0
    for row_idx in indices:
        if len(items) >= n_items:
            break
        row = ds[row_idx]
        question = (row["question"] or "").strip()
        best = (row["best_answer"] or "").strip()
        incorrect = [a.strip() for a in (row["incorrect_answers"] or []) if a and a.strip()]
        if not question or not best or _ill_posed(question, best):
            continue

        # item_id is opaque (a running index), so the ground-truth label is not
        # encoded in the identifier; the answer field is retained in the
        # provenance record for traceability, not in the id a consumer sorts on.
        def _mk(answer: str, label: str, which: str) -> SourceItem:
            nonlocal seq
            seq += 1
            text = f"Q: {question}\nA: {answer}"
            return SourceItem(
                item_id=f"fact_tqa_{seq:04d}",
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

# A retrieval corpus can carry records whose body text is missing. In
# BeIR/trec-covid, 42,607 of 171,332 documents (25%) have an absent abstract,
# stored as the literal string "Unknown" or as a fragment too short to judge.
# Such a document is still a valid TREC judgement, so it survives every
# label-level check, but it cannot support a relevance decision: a judge picks
# against it by noticing it is empty, not by assessing relevance. One shipped in
# v2 (item relv_treccovid_17_0, whose non-relevant candidate was "Unknown").
# Excluded from the candidate pool at source so it can be selected neither as a
# positive nor as a distractor.
_PLACEHOLDER_DOC = re.compile(
    r"^(unknown|none|n/?a|null|nan|untitled|no title|no abstract)$", re.IGNORECASE
)
_MIN_DOCUMENT_CHARS = 60


def _is_usable_document(text: str, min_chars: int = _MIN_DOCUMENT_CHARS) -> bool:
    """A document substantial enough that relevance is judged by reading it.

    `min_chars` is a parameter so a test exercising unrelated logic (BM25
    ranking, qrels grading) can use compact fixtures. The shipped default is
    deliberately strict; lowering it in a real build reintroduces documents a
    judge answers against by noticing they are empty.
    """
    t = (text or "").strip()
    return bool(t) and len(t) >= min_chars and not _PLACEHOLDER_DOC.match(t)


def load_relevance_items(
    n_items: int = DEFAULT_ITEMS_PER_TASK,
    seed: int = 42,
    corpus_id: str = "BeIR/trec-covid",
    qrels_id: str = "BeIR/trec-covid-qrels",
    qrels_split: str = "test",
    difficulty: str = "hard",
    min_document_chars: int = _MIN_DOCUMENT_CHARS,
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
    n_excluded_unusable = 0
    for row in corpus:
        did = str(row["_id"])
        if did not in judged_ids:
            continue
        text = (row["text"] or "").strip()
        if not _is_usable_document(text, min_document_chars):
            n_excluded_unusable += 1
            continue
        corpus_by_id[did] = text

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
    # Headroom above the even split so that topics with many graded documents
    # can cover a shortfall left by topics with few (lexical matching can leave
    # a sparse topic unable to fill its even share).
    per_query_cap = max(1, math.ceil(n_items / len(qids)) + 4)

    def _texts_ok(a: str, b: str) -> bool:
        return bool(a) and bool(b) and a != b

    retrieved_at = _now_iso()
    items: List[SourceItem] = []
    used_pairs: set = set()
    n_excluded_duplicate = 0

    # BM25 of the query against each candidate. Critically, the negative is
    # chosen to be LEXICALLY MATCHED to the positive, not maximally overlapping.
    #
    # Selecting the highest-BM25 explicit negative (an earlier version) made the
    # non-relevant document keyword-denser than the truly relevant one, so
    # "pick the passage with LOWER query-term overlap" recovered relevance ~75%+
    # of the time -- a lexical shortcut in reverse. Matching the negative's BM25
    # to the positive's removes overlap as a signal in either direction, so a
    # judge must read for relevance rather than count shared terms.
    q_pos_scores: Dict[str, Dict[str, float]] = {}
    q_neg_scores: Dict[str, Dict[str, float]] = {}
    for qid in qids:
        scores = index.scores(query_by_id.get(qid, ""))
        idx_of = {index.doc_ids[i]: i for i in range(index.n_docs)}
        def _sc(did: str) -> float:
            i = idx_of.get(did)
            return scores.get(i, 0.0) if i is not None else 0.0
        q_pos_scores[qid] = {d: _sc(d) for d in positives[qid] if corpus_by_id.get(d)}
        q_neg_scores[qid] = {d: _sc(d) for d in negatives[qid]
                             if d in idx_of and corpus_by_id.get(d)}

    used_neg: Dict[str, set] = defaultdict(set)
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
            pos_list = sorted(q_pos_scores[qid])
            avail = [(nid, s) for nid, s in q_neg_scores[qid].items()
                     if nid not in used_neg[qid]]
            if not pos_list or not avail:
                continue
            pos_id = pos_list[j % len(pos_list)]
            pos_score = q_pos_scores[qid][pos_id]
            if difficulty == "hard":
                # lexically matched: negative whose BM25 is closest to the positive's.
                neg_id, neg_score = min(avail, key=lambda t: (abs(t[1] - pos_score), t[0]))
            else:
                neg_id, neg_score = avail[rng.randrange(len(avail))]
            used_neg[qid].add(neg_id)
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
                                "non-relevant=TREC grade 0; distractor is the explicit "
                                "negative whose query BM25 is closest to the positive's, "
                                "so lexical overlap does not identify the relevant doc"
                            ),
                            "difficulty": difficulty,
                            "pos_bm25_score": f"{pos_score:.4f}",
                            "neg_bm25_score": f"{neg_score:.4f}",
                            "bm25_abs_gap": f"{abs(neg_score - pos_score):.4f}",
                            "n_explicit_negatives": str(len(q_neg_scores[qid])),
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
    length_balance: bool = True,
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

    ``length_balance`` (default True) holds winner-longer at exactly 50% AND
    matches the two length buckets on vote margin. Because MT-Bench supplies
    only 113 winner-shorter comparisons against 272 winner-longer ones, a trim
    that took the most-contested 113 of the longer bucket while keeping the
    shorter bucket whole left the buckets matched on length but systematically
    UNMATCHED on annotator agreement (measured: mean margin ratio 0.567 longer
    vs 0.745 shorter; 16% vs 54% unanimous). Length was then decorrelated from
    the label at the price of correlating it with how contested the comparison
    is — a judge good on close calls reads as length-biased, and vice versa.
    The selection is therefore STRATIFIED on margin: the smaller bucket fixes
    the target margin distribution and the larger bucket is filled stratum by
    stratum to match it, so neither length nor agreement carries information
    about the label. ``difficulty`` still orders the smaller bucket when the
    request is smaller than the pool, and breaks ties within a stratum.
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

    margins = {key: _margin(key) for key in keys}

    def _winner_longer(key) -> bool:
        """True if the human-preferred response is the longer one. Used to
        BALANCE length, not to select on it: MT-Bench annotators preferred the
        longer answer ~69% of the time, so an unbalanced pool lets a judge score
        ~69% by always picking the longer response (verbosity bias). Balancing
        winner-longer to ~50% removes length as a usable signal."""
        _q, ra, rb, _r = texts[key]
        directional = {k: v for k, v in votes[key].items() if k in ("model_a", "model_b")}
        winners = [k for k, v in directional.items() if v == max(directional.values())]
        winner_text = ra if winners[0] == "model_a" else rb
        loser_text = rb if winners[0] == "model_a" else ra
        return len(winner_text) > len(loser_text)

    def _contest_key(k):
        m = margins[k]
        return (m[2], m[0]) if difficulty == "hard" else (-m[2], -m[0])

    eligible = [k for k in keys if margins[k] is not None]
    longer = sorted([k for k in eligible if _winner_longer(k)], key=_contest_key)
    shorter = sorted([k for k in eligible if not _winner_longer(k)], key=_contest_key)
    # Interleave the two length buckets so the winner is longer in ~half the
    # selected items; within each bucket the most contested comparisons come
    # first. A tail from the larger bucket is taken only if one bucket is
    # exhausted before n_items is reached; the residual balance is reported.
    # Produce the FULL balanced ordering of all eligible comparisons, not just
    # n_items of them: the item-building loop below drops duplicate-content
    # pairs, so it needs surplus to still reach n_items. Taking the first n_items
    # of this balanced order keeps winner-longer near 50%.
    keys = []
    li = si = 0
    while li < len(longer) or si < len(shorter):
        take_longer = li < len(longer) and (si >= len(shorter) or li <= si)
        if take_longer:
            keys.append(longer[li]); li += 1
        else:
            keys.append(shorter[si]); si += 1

    retrieved_at = _now_iso()
    items: List[SourceItem] = []
    n_excluded_tie = 0
    n_excluded_duplicate = 0
    n_excluded_length_quota = 0
    seen_content: set = set()
    # Balancing the ORDER the candidates are considered in is not enough to
    # balance the shipped set: the tie, empty-text and duplicate-content filters
    # below drop items unevenly, so the interleave decays. Measured on the v2
    # build, "always pick the longer response" still scored 0.548-0.560 -- a
    # residual verbosity shortcut. A hard per-bucket quota makes the shipped
    # split exactly even regardless of what the filters remove, so length
    # carries no information about the label.
    bucket_of: Dict[str, bool] = {}
    # (margin ratio, absolute margin, total votes) per shipped item, so the trim
    # below can stratify the two length buckets on annotator agreement as well
    # as on length.
    margin_of: Dict[str, Tuple[float, int, int]] = {}
    for key in keys:
        # No early break at n_items: the whole eligible pool is collected so the
        # balanced trim below can choose from all of it.
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
        winner_is_longer = _winner_longer(key)

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
                        "contested": "yes" if margin_ratio < 1.0 else "no (unanimous majority)",
                        "vote_margin": str(margin),
                        "vote_margin_ratio": f"{margin_ratio:.4f}",
                        "total_votes": str(total_votes),
                        "winner_is_longer": "yes" if len(ra if label == "candidate_1" else rb) > len(rb if label == "candidate_1" else ra) else "no",
                        "winner_chars": str(len(ra if label == "candidate_1" else rb)),
                        "loser_chars": str(len(rb if label == "candidate_1" else ra)),
                    },
                    retrieved_at=retrieved_at,
                ),
                extra={"candidate_1": ra, "candidate_2": rb},
            )
        )
        bucket_of[items[-1].item_id] = winner_is_longer
        margin_of[items[-1].item_id] = (margin_ratio, margin, total_votes)

    # Exact length balance AND margin balance, chosen over hitting a round
    # item count.
    #
    # MT-Bench annotators preferred the longer response in 272 of the 385 usable
    # pairs (70.6%), so "always pick the longer response" is a shortcut that
    # scores far above chance on an unbalanced sample. Ordering the candidates by
    # length bucket is not sufficient, because the tie, empty-text and
    # duplicate-content filters remove items unevenly and the interleave decays:
    # the v2 build shipped 250 items at 54.8% winner-longer, still an exploitable
    # signal. Taking an equal number from each bucket makes it exactly 50%, so
    # length carries no information about the label.
    #
    # Taking the FIRST per_bucket of each bucket in most-contested-first order,
    # however, applies that selection asymmetrically. The shorter bucket has
    # exactly 113 members, so all of them survive unselected; the longer bucket
    # has 272, so only its most contested 113 survive. Measured on this pool the
    # two buckets then differ sharply in annotator agreement — mean vote-margin
    # ratio 0.5665 (median 0.500, 18/113 unanimous) for winner-longer against
    # 0.7453 (median 1.000, 61/113 unanimous) for winner-shorter — which trades
    # a length confound for an agreement confound: a judge that is good on close
    # calls scores better on the longer bucket and reads as length-biased.
    #
    # The fix stratifies. The bucket that binds (the smaller one, or either one
    # when n_items caps both) is selected first and fixes the target margin
    # distribution; the other bucket is then filled stratum by stratum to match
    # it, so the two buckets are matched on margin as well as on length. Strata
    # are exact vote shapes (absolute margin, total votes), so the match covers
    # the margin ratio, the raw margin and the annotation depth at once. On this
    # pool 111 of 113 match exactly; the two 5-0 and 6-0 winner-shorter
    # comparisons have no winner-longer counterpart at that vote shape and are
    # filled from the nearest-ratio stratum.
    #
    # The smaller bucket still caps the split at 2 x 113 = 226 items rather
    # than the requested 250. That is a real limit of the human-labelled pool,
    # not padding: the alternative is to ship 24 more items that reintroduce a
    # measurable verbosity shortcut across the whole task. Note that the trim
    # DISCARDS 159 eligible majority-labelled comparisons (385 - 226), not 24:
    # 24 is only the gap between the request and what ships.
    longer_items = [it for it in items if bucket_of[it.item_id]]
    shorter_items = [it for it in items if not bucket_of[it.item_id]]
    n_eligible = len(items)

    def _match_margin_distribution(pool: List[SourceItem],
                                   reference: List[SourceItem]) -> List[SourceItem]:
        """Pick len(reference) items from `pool` whose vote margins have the
        same distribution as `reference`'s.

        A stratum is the exact vote shape (absolute margin, total votes), which
        also fixes the margin ratio — so matching strata matches the ratio, the
        raw margin and the annotation depth together. Within a stratum `pool` is
        already in difficulty order, so `difficulty` breaks ties. A stratum the
        pool cannot fill is topped up from the stratum with the nearest margin
        ratio (on a tie, the more contested side under "hard" and the more
        decisive side under "easy"), which keeps the count exact when the pool's
        support does not cover the reference's.
        """
        def _stratum(it: SourceItem) -> Tuple[int, int]:
            _ratio, margin_, total_ = margin_of[it.item_id]
            return (margin_, total_)

        need = Counter(_stratum(it) for it in reference)
        available: Dict[Tuple[int, int], List[SourceItem]] = defaultdict(list)
        for it in pool:
            available[_stratum(it)].append(it)
        chosen: List[SourceItem] = []
        deficit: Counter = Counter()
        for stratum in sorted(need, reverse=True):
            take = min(need[stratum], len(available[stratum]))
            chosen.extend(available[stratum][:take])
            available[stratum] = available[stratum][take:]
            if take < need[stratum]:
                deficit[stratum] = need[stratum] - take
        for stratum in sorted(deficit, reverse=True):
            want_ratio = stratum[0] / stratum[1]
            for _ in range(deficit[stratum]):
                open_strata = [s for s, v in available.items() if v]
                if not open_strata:
                    break
                nearest = min(open_strata, key=lambda s: (
                    abs(s[0] / s[1] - want_ratio),
                    (s[0] / s[1]) if difficulty == "hard" else -(s[0] / s[1]),
                    s,
                ))
                chosen.append(available[nearest].pop(0))
        return chosen

    if length_balance:
        per_bucket = min(len(longer_items), len(shorter_items), n_items // 2)
        n_excluded_length_quota = n_eligible - 2 * per_bucket
        # The bucket with less headroom fixes the margin distribution; the other
        # is matched to it. Ties go to the shorter bucket, which is the scarce
        # one on every real MT-Bench build.
        if len(shorter_items) <= len(longer_items):
            reference, pool = shorter_items[:per_bucket], longer_items
        else:
            reference, pool = longer_items[:per_bucket], shorter_items
        keep = {it.item_id for it in reference}
        keep |= {it.item_id for it in _match_margin_distribution(pool, reference)}
        items = [it for it in items if it.item_id in keep]
        # A shortfall the balance itself caused is a deliberate trade, reported
        # rather than raised. A shortfall the POOL caused is still fatal: too
        # few eligible pairs to fill the request means the source could not
        # supply the benchmark, and padding it is the v1 defect.
        balance_limited = 2 * per_bucket < n_items <= n_eligible
    else:
        # Only for tests exercising margin/tie logic on fixtures too small to
        # balance. A build with this off ships the pool's native ~70% verbosity
        # skew, which is the shortcut the balance exists to remove.
        per_bucket = n_items // 2
        n_excluded_length_quota = 0
        items = items[:n_items]
        balance_limited = False

    if len(items) < n_items and not balance_limited:
        raise DataSourceSchemaError(
            f"MT-Bench human judgments yielded only {n_eligible} usable "
            f"majority-labeled pairs (requested {n_items}, "
            f"{n_excluded_tie} excluded as tie/no-majority, "
            f"{n_excluded_duplicate} as duplicate content); refusing to pad."
        )
    if balance_limited:
        kept_longer = [it for it in items if bucket_of[it.item_id]]
        kept_shorter = [it for it in items if not bucket_of[it.item_id]]
        mean_l = sum(margin_of[it.item_id][0] for it in kept_longer) / max(len(kept_longer), 1)
        mean_s = sum(margin_of[it.item_id][0] for it in kept_shorter) / max(len(kept_shorter), 1)
        print(
            f"  [preference] length-balanced to {len(items)} items "
            f"({per_bucket} winner-longer + {per_bucket} winner-shorter); "
            f"{n_excluded_length_quota} of {n_eligible} eligible majority-labeled "
            f"pairs dropped to hold the balance exactly at 50%. "
            f"Requested {n_items}."
        )
        print(
            f"  [preference] margin-matched buckets: mean vote-margin ratio "
            f"{mean_l:.4f} (winner-longer) vs {mean_s:.4f} (winner-shorter), "
            f"delta {abs(mean_l - mean_s):.4f}."
        )
    return items


TASK_LOADERS = {
    "factuality": load_factuality_items,
    "coherence": load_coherence_items,
    "relevance": load_relevance_items,
    "preference": load_preference_items,
}
