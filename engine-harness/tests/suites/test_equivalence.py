"""Greedy outputs must match the phase-01 reference: optimizations may not change answers."""

import pytest

from engineharness.constants import SUITE_EQUIVALENCE, WORKLOAD_EQUIVALENCE
from engineharness.output_equivalence import collect_greedy_outputs, compare_outputs, load_reference
from engineharness.report_writer import SuiteResult
from engineharness.workloads import build_equivalence_requests

pytestmark = pytest.mark.anyio


@pytest.mark.equivalence
async def test_greedy_outputs_match_the_reference(
    request, harness_settings, harness_report, engine_session, engine_tokenizer
) -> None:
    if not harness_settings.reference_file.exists():
        pytest.skip(f"no reference yet at {harness_settings.reference_file}")
    reference = load_reference(harness_settings.reference_file)
    requests = build_equivalence_requests(
        engine_tokenizer,
        harness_settings.equivalence_prompt_count,
        harness_settings.equivalence_output_tokens,
        harness_settings.seed,
    )

    outputs = await collect_greedy_outputs(engine_session, requests, harness_settings.model_id)
    result = compare_outputs(reference, outputs)

    harness_report.record(
        request.node.nodeid,
        SuiteResult(
            SUITE_EQUIVALENCE,
            WORKLOAD_EQUIVALENCE,
            f"{result.identical_count}/{result.prompt_count} identical "
            f"(prefix match {result.mean_common_prefix_ratio:.0%})",
            f">= {harness_settings.min_equivalence_agreement:.0%}",
            ", ".join(result.mismatched_request_ids) or "all match",
        ),
    )
    assert result.agreement >= harness_settings.min_equivalence_agreement
