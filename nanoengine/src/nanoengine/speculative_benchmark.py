"""Single-user speed with and without speculative decoding:
``python -m nanoengine.speculative_benchmark --output-dir results``.

Uses the configured ``speculative`` proposer and target model. Two prompt styles: a code edit
(the answer copies much of the prompt, where guesses are often right) and free chat text.
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from nanoengine.config import EngineSettings, load_engine_settings
from nanoengine.constants import TORCH_DTYPES
from nanoengine.engine.generation.cached_generator import CachedGenerator
from nanoengine.engine.generation.model_runner import ModelRunner
from nanoengine.engine.interface import GenerationRequest, SamplingParams
from nanoengine.engine.memory.contiguous_allocator import ContiguousSlotAllocator
from nanoengine.engine.memory.kv_cache_pool import KVCachePool
from nanoengine.engine.metrics import EngineMetrics
from nanoengine.engine.runtime import create_proposer
from nanoengine.engine.speculative.speculative_generator import SpeculativeGenerator
from nanoengine.models.qwen3 import Qwen3ForCausalLM
from nanoengine.models.text_tokenizer import TextTokenizer, load_text_tokenizer
from nanoengine.models.weight_loader import download_model_files, load_qwen3_model

PROMPT_STYLES = {
    "code_edit": (
        "def total_price(items, tax_rate):\n"
        "    subtotal = 0\n"
        "    for item in items:\n"
        "        subtotal += item.price * item.quantity\n"
        "    tax = subtotal * tax_rate\n"
        "    return subtotal + tax\n\n"
        "# The same function with the variable subtotal renamed to net_total:\n"
        "def total_price(items, tax_rate):\n"
    ),
    "chat": (
        "Explain in a few sentences why a city might build more bicycle lanes, and what "
        "residents usually say about it."
    ),
}
GENERATED_TOKENS = 128
REPEATS = 3
SPECULATIVE_FILE_STEM = "speculative-benchmark"


@dataclass(frozen=True)
class SpeculativeResult:
    prompt_style: str
    baseline_tokens_per_second: float
    speculative_tokens_per_second: float
    speed_up: float
    acceptance_rate: float
    tokens_per_target_pass: float


def build_runner(model: Qwen3ForCausalLM, settings: EngineSettings) -> ModelRunner:
    device = torch.device(settings.device)
    pool = KVCachePool(model.config, settings.max_model_len, device, TORCH_DTYPES[settings.dtype])
    allocator = ContiguousSlotAllocator(1, settings.max_model_len, device)
    return ModelRunner(model, pool, allocator, device)


def time_generation(generator, prompt: list[int]) -> tuple[float, int]:
    """Seconds and tokens for ``REPEATS`` greedy generations of ``GENERATED_TOKENS`` tokens."""
    params = SamplingParams(max_new_tokens=GENERATED_TOKENS, ignore_eos=True)
    started_at = time.perf_counter()
    num_tokens = 0
    for repeat in range(REPEATS):
        request = GenerationRequest(f"bench-{repeat}", prompt, params)
        num_tokens += sum(1 for _ in generator.generate(request))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - started_at, num_tokens


def run_benchmark(
    model: Qwen3ForCausalLM,
    draft_model: Qwen3ForCausalLM | None,
    tokenizer: TextTokenizer,
    settings: EngineSettings,
) -> list[SpeculativeResult]:
    device = torch.device(settings.device)
    baseline = CachedGenerator(build_runner(model, settings), frozenset(), device)
    results = []
    for style, text in PROMPT_STYLES.items():
        metrics = EngineMetrics()
        speculative = SpeculativeGenerator(
            build_runner(model, settings),
            create_proposer(settings, draft_model),
            settings.num_speculative_tokens,
            frozenset(),
            device,
            metrics,
        )
        prompt = tokenizer.encode(text)
        time_generation(baseline, prompt)  # warm-up
        baseline_seconds, baseline_tokens = time_generation(baseline, prompt)
        speculative_seconds, speculative_tokens = time_generation(speculative, prompt)
        proposed = metrics.registry.get_sample_value("nanoengine_speculative_proposed_tokens_total")
        accepted = metrics.registry.get_sample_value("nanoengine_speculative_accepted_tokens_total")
        target_passes = speculative_tokens - accepted
        results.append(
            SpeculativeResult(
                prompt_style=style,
                baseline_tokens_per_second=baseline_tokens / baseline_seconds,
                speculative_tokens_per_second=speculative_tokens / speculative_seconds,
                speed_up=baseline_seconds / speculative_seconds,
                acceptance_rate=accepted / proposed if proposed else 0.0,
                tokens_per_target_pass=speculative_tokens / max(target_passes, 1),
            )
        )
    return results


def render_results_markdown(results: list[SpeculativeResult], settings: EngineSettings) -> str:
    lines = [
        f"# Speculative decoding: {settings.speculative}, k={settings.num_speculative_tokens}, "
        f"target {settings.model_id}",
        "",
        "| Prompt | Baseline tok/s | Speculative tok/s | Speed-up | Acceptance | Tokens per pass |",
        "|---|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {result.prompt_style} | {result.baseline_tokens_per_second:.1f} "
        f"| {result.speculative_tokens_per_second:.1f} | {result.speed_up:.2f}x "
        f"| {result.acceptance_rate:.0%} | {result.tokens_per_target_pass:.2f} |"
        for result in results
    )
    return "\n".join(lines) + "\n"


def main(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="nanoengine.speculative_benchmark")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parsed = parser.parse_args(arguments)
    settings = load_engine_settings()
    device = torch.device(settings.device)
    model_dir = download_model_files(settings.model_id, settings.model_cache_dir)
    model = load_qwen3_model(model_dir, device, TORCH_DTYPES[settings.dtype])
    results = run_benchmark(model, None, load_text_tokenizer(model_dir), settings)
    parsed.output_dir.mkdir(parents=True, exist_ok=True)
    stem = parsed.output_dir / SPECULATIVE_FILE_STEM
    stem.with_suffix(".md").write_text(render_results_markdown(results, settings), "utf-8")
    stem.with_suffix(".json").write_text(
        json.dumps([asdict(result) for result in results], indent=2) + "\n", "utf-8"
    )


if __name__ == "__main__":
    main()
