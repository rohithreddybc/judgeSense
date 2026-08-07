"""
Tests for src/structural_variants.py and the structural-axis metrics.

Raw strings here are TEST FIXTURES (labelled probe outputs), never model
outputs, and are never written under data/.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics_v2 import (  # noqa: E402
    MetricContractError,
    assert_jss_eligible,
    cluster_bootstrap_ci,
    format_failure_rate,
    jss,
    mean_likert_shift,
    structural_shift_rate,
)
from src.structural_variants import (  # noqa: E402
    LABEL_SETS,
    UNCLEAR,
    VARIANT_CLASS,
    VARIANT_IDS,
    VariantError,
    assert_no_polarity_drift,
    enumerate_pairs,
    json_label_map,
    parse_variant_output,
    render_variant,
    split_instruction,
    structural_pair_id,
)

POINTWISE = "Is this statement factually correct? Answer YES or NO only.\n\nThe Earth orbits the Sun."
PAIRWISE = 'Which passage is more relevant to the query "x"? Answer A or B only.\nA: first\nB: second'


# ── rendering ────────────────────────────────────────────────────────────────

def test_s0_is_unchanged_user_only():
    out = render_variant("factuality", POINTWISE, "S0")
    assert out["system"] is None
    assert out["user"] == POINTWISE
    assert out["parse_mode"] == "plain"


def test_every_variant_renders_for_every_task():
    for task, probe in [("factuality", POINTWISE), ("relevance", PAIRWISE)]:
        for variant in VARIANT_IDS:
            out = render_variant(task, probe, variant)
            assert out["user"].strip()
            assert out["variant_class"] == VARIANT_CLASS[variant]


def test_s2_relocates_instruction_and_keeps_body():
    out = render_variant("factuality", POINTWISE, "S2")
    assert out["system"] == "Is this statement factually correct? Answer YES or NO only."
    assert out["user"] == "The Earth orbits the Sun."


def test_s2_pairwise_keeps_query_with_instruction():
    # The query is part of what is being asked, not one of the candidates.
    out = render_variant("relevance", PAIRWISE, "S2")
    assert "query" in out["system"]
    assert out["user"].startswith("A: first")
    assert "B: second" in out["user"]


def test_s1_label_map_is_identity_bijection():
    for task in LABEL_SETS:
        mapping = json_label_map(task)
        assert set(mapping.keys()) == set(LABEL_SETS[task])
        assert set(mapping.values()) == set(LABEL_SETS[task])


def test_unknown_task_or_variant_raises():
    with pytest.raises(VariantError):
        render_variant("nope", POINTWISE, "S0")
    with pytest.raises(VariantError):
        render_variant("factuality", POINTWISE, "S9")


def test_split_instruction_rejects_unsplittable_prompt():
    with pytest.raises(VariantError):
        split_instruction("factuality", "no blank line here")
    with pytest.raises(VariantError):
        split_instruction("relevance", "no candidates line")


# ── polarity guard (the v1 Template-4 defect class) ──────────────────────────

def test_no_variant_introduces_polarity_drift():
    for task in LABEL_SETS:
        assert_no_polarity_drift(task)


# ── parsing ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ('{"verdict": "YES"}', "YES"),
    ('```json\n{"verdict": "NO"}\n```', "NO"),
    ('{"verdict": "MAYBE"}', UNCLEAR),           # outside the label space
    ('{"verdict": "YES", "why": "..."}', UNCLEAR),  # extra keys break the contract
    ("YES", UNCLEAR),                            # not JSON at all
    ("", UNCLEAR),
])
def test_json_parse_mode(raw, expected):
    assert parse_variant_output("factuality", raw, "json") == expected


@pytest.mark.parametrize("raw,expected", [
    ("reasoning here\nFINAL: YES", "YES"),
    ("step 1\nstep 2\nfinal: NO", "NO"),
    ("FINAL: YES\nFINAL: NO", "NO"),      # last marker wins
    ("lots of reasoning, no marker", UNCLEAR),
])
def test_final_marker_parse_mode(raw, expected):
    assert parse_variant_output("factuality", raw, "final_marker") == expected


@pytest.mark.parametrize("raw,expected", [
    ("B", "B"),
    ("**B**", "B"),
    ("Answer: B", "B"),
    # v1 parsed these as "A" via the uppercased English article.
    ("This is a tough call, but B", "B"),
    ("As a judge, I select B", "B"),
    # Naming both candidates is not evidence of either.
    ("Could be A or B", UNCLEAR),
    ("Neither passage answers the query", UNCLEAR),
])
def test_plain_parse_mode_resists_v1_article_bug(raw, expected):
    assert parse_variant_output("relevance", raw, "plain") == expected


@pytest.mark.parametrize("raw,expected", [
    ("4", "4"),
    ("Score: 4", "4"),
    # v1 grabbed the first digit anywhere, returning 1 from the scale echo.
    ("On a scale of 1-5, I'd say 4", UNCLEAR),
    ("32", UNCLEAR),
])
def test_plain_parse_mode_coherence(raw, expected):
    assert parse_variant_output("coherence", raw, "plain") == expected


# ── pair enumeration ─────────────────────────────────────────────────────────

def test_enumerate_pairs_is_a_star_at_s0():
    pairs = enumerate_pairs("item_7")
    assert len(pairs) == 5
    assert {p["variant_a"] for p in pairs} == {"S0"}
    assert {p["variant_b"] for p in pairs} == {"S1", "S2", "S3", "S4", "S5"}
    assert pairs[0]["structural_pair_id"] == structural_pair_id("item_7", "S1")


# ── metric contract: Class N must never be scored as JSS ─────────────────────

def _rec(a, b, item="i0", variant="S1", klass="E"):
    return {
        "decision_a": a, "decision_b": b,
        "item_id": item,
        "structural_pair_id": structural_pair_id(item, variant),
        "variant_class": klass,
    }


def test_jss_refuses_class_n_records():
    recs = [_rec("YES", "NO", f"i{i}", "S4", "N") for i in range(5)]
    with pytest.raises(MetricContractError, match="Class N"):
        assert_jss_eligible(recs)


def test_jss_allows_class_e_and_bare_instruction_axis_records():
    assert_jss_eligible([_rec("YES", "YES", f"i{i}") for i in range(5)])
    assert_jss_eligible([{"decision_a": "YES", "decision_b": "YES"}])


def test_structural_shift_rate_reports_direction_not_just_magnitude():
    # An intervention that systematically pushes YES -> NO.
    recs = [_rec("YES", "NO", f"i{i}", "S4", "N") for i in range(8)]
    recs += [_rec("NO", "NO", f"i{i}", "S4", "N") for i in range(8, 12)]
    out = structural_shift_rate(recs)
    assert out["structural_shift_rate"] == pytest.approx(8 / 12)
    assert out["transitions"]["YES->NO"] == 8
    assert out["net_flow"]["NO"] == 8
    assert out["net_flow"]["YES"] == -8


def test_mean_likert_shift_is_signed():
    harsher = [_rec("4", "3", f"i{i}", "S5", "N") for i in range(10)]
    out = mean_likert_shift(harsher)
    assert out["mean_shift"] == pytest.approx(-1.0)
    assert out["harsher"] == 10 and out["more_lenient"] == 0


def test_mean_likert_shift_counts_unparseable_separately():
    recs = [_rec("4", "3", f"i{i}", "S5", "N") for i in range(5)]
    recs += [_rec("4", UNCLEAR, f"u{i}", "S5", "N") for i in range(3)]
    out = mean_likert_shift(recs)
    assert out["n_scored"] == 5 and out["n_unparseable"] == 3


def test_format_failure_rate_counts_unclear_arm_outputs():
    recs = [_rec("YES", "YES", f"i{i}") for i in range(8)]
    recs += [_rec("YES", UNCLEAR, f"u{i}") for i in range(2)]
    out = format_failure_rate(recs)
    assert out["format_failure_rate"] == pytest.approx(0.2)
    assert out["n_failed"] == 2


# ── clustering on the structural axis ────────────────────────────────────────

def test_structural_pair_is_a_valid_cluster_unit():
    recs = [_rec("YES", "YES", f"i{i}", "S1") for i in range(20)]
    out = cluster_bootstrap_ci(recs, lambda r: jss(r, "disagree"),
                               cluster_unit="structural_pair", n_bootstrap=50)
    assert out["cluster_unit"] == "structural_pair"
    assert out["n_clusters"] == 20


def test_item_clustering_is_wider_when_pairs_share_an_item():
    # Five structural pairs per item all share that item's S0 arm, so item-level
    # resampling must not treat them as five independent observations.
    recs = []
    for i in range(20):
        agree = i % 2 == 0
        for variant in ("S1", "S2", "S3"):
            recs.append(_rec("YES", "YES" if agree else "NO", f"i{i}", variant))
    by_pair = cluster_bootstrap_ci(recs, lambda r: jss(r, "disagree"),
                                   cluster_unit="structural_pair", n_bootstrap=400)
    by_item = cluster_bootstrap_ci(recs, lambda r: jss(r, "disagree"),
                                   cluster_unit="item", n_bootstrap=400)
    assert by_pair["n_clusters"] == 60 and by_item["n_clusters"] == 20
    width_pair = by_pair["ci_upper"] - by_pair["ci_lower"]
    width_item = by_item["ci_upper"] - by_item["ci_lower"]
    assert width_item > width_pair, (
        "item-level CI must be wider when pairs are nested within items; "
        "otherwise the v1 independence error recurs one level up"
    )
