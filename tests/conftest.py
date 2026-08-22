"""
Shared pytest configuration.

Marks the v1 analysis tests as expected failures rather than leaving them
permanently red. They assert properties of the v1 experiment -- per-template JSS,
pair-level flip overlap, and the Template-4 polarity audit -- against
`data/results/raw_outputs`, which a later partial re-run overwrote with outputs
whose prompts no longer match the v1 templates those tests search for.

They are kept rather than deleted because the T4 polarity finding they encode is
cited in the paper as a benchmark-design lesson, and the tests document how it
was originally measured. They are marked because a suite that is always red
teaches everyone to ignore it, which is how a real regression gets shipped: the
seven failures were treated as background noise for most of this project, and a
genuine budget-arithmetic regression had to be caught by hand instead.

An xfail that starts PASSING is reported as XPASS, so if the v1 artifacts are
ever restored these tests speak up rather than staying silent.
"""

import pytest

# Exactly the assertions that depend on v1 raw outputs which no longer exist in
# the form they expect. Listed individually rather than by module: most tests in
# these files still pass, and blanket-marking a module would hide any of them
# that later broke for a real reason.
_V1_ARTIFACT_TESTS = {
    "test_factuality_jss::test_all_models_present",
    "test_factuality_jss::test_delta_equals_difference",
    "test_pair_overlap::test_flip_count_in_range",
    "test_per_template::test_t4_all_models",
    "test_t4_audit::test_t4_pairs_identified",
    "test_t4_audit::test_t4_stored_raw_not_corrected",
    "test_t4_audit::test_inversion_improves_t3t4_agreement",
}

_REASON = (
    "v1 analysis test: asserts against data/results/raw_outputs, which a later "
    "partial re-run overwrote with outputs whose prompts no longer match the v1 "
    "templates. Superseded by the v2 pipeline; retained because the T4 polarity "
    "finding it measures is cited in the paper."
)


def pytest_collection_modifyitems(config, items):
    for item in items:
        module = item.module.__name__.rsplit(".", 1)[-1]
        # `originalname` so a parametrised case matches on its base name.
        name = getattr(item, "originalname", None) or item.name.split("[")[0]
        if f"{module}::{name}" in _V1_ARTIFACT_TESTS:
            # strict: an unexpected PASS fails the suite, so restoring the v1
            # artifacts forces someone to remove the marker rather than letting
            # a stale exemption sit here forever.
            item.add_marker(pytest.mark.xfail(reason=_REASON, strict=True))
