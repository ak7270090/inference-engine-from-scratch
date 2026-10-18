"""Command line: ``python -m engineharness {wait,sweep,reference}``."""

import argparse
import asyncio
import sys

from engineharness.client_factory import create_http_session, wait_until_ready
from engineharness.concurrency_sweep import run_concurrency_sweep
from engineharness.config import HarnessSettings, load_harness_settings
from engineharness.output_equivalence import collect_greedy_outputs, save_reference
from engineharness.report_writer import write_sweep_results
from engineharness.workloads import (
    HubTokenizer,
    build_equivalence_requests,
    build_requests,
    get_workload_spec,
)


async def wait_for_engine(settings: HarnessSettings) -> int:
    async with create_http_session(settings) as session:
        is_ready = await wait_until_ready(session, settings.server_ready_timeout_seconds)
    print("engine ready" if is_ready else "engine not ready in time")
    return 0 if is_ready else 1


async def sweep_workload(settings: HarnessSettings, workload_name: str) -> int:
    spec = get_workload_spec(settings.workload_scale, workload_name)
    requests = build_requests(spec, HubTokenizer(settings.tokenizer_id), settings.seed)
    async with create_http_session(settings) as session:
        metrics = await run_concurrency_sweep(
            session, requests, settings.concurrency_levels, settings.model_id, workload_name
        )
    for path in write_sweep_results(settings.results_dir, metrics, settings.target_url):
        print(f"wrote {path}")
    return 0 if all(level.error_count == 0 for level in metrics) else 1


async def write_reference(settings: HarnessSettings) -> int:
    requests = build_equivalence_requests(
        HubTokenizer(settings.tokenizer_id),
        settings.equivalence_prompt_count,
        settings.equivalence_output_tokens,
        settings.seed,
    )
    async with create_http_session(settings) as session:
        outputs = await collect_greedy_outputs(session, requests, settings.model_id)
    save_reference(settings.reference_file, outputs, source=settings.target_url)
    print(f"wrote {len(outputs)} reference outputs to {settings.reference_file}")
    return 0


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="engineharness")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("wait", help="wait until the engine answers /v1/models")
    sweep_parser = commands.add_parser("sweep", help="run a workload at every concurrency level")
    sweep_parser.add_argument("workload", help="short_chat, long_prompt, shared_prefix, ...")
    commands.add_parser("sweep-all", help="sweep every workload in HARNESS_SWEEP_WORKLOADS")
    commands.add_parser("reference", help="save greedy outputs as the equivalence reference")
    parsed = parser.parse_args(arguments)

    settings = load_harness_settings()
    if parsed.command == "wait":
        return asyncio.run(wait_for_engine(settings))
    if parsed.command == "sweep":
        return asyncio.run(sweep_workload(settings, parsed.workload))
    if parsed.command == "sweep-all":
        exit_codes = [
            asyncio.run(sweep_workload(settings, workload)) for workload in settings.sweep_workloads
        ]
        return max(exit_codes, default=0)
    return asyncio.run(write_reference(settings))


if __name__ == "__main__":
    sys.exit(main())
