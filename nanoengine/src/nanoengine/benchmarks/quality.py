"""What int8 costs in quality: ``python -m nanoengine.benchmarks.quality --output-dir results``.

Compares the configured engine (``quantization``, ``kv_cache_dtype``) against the same model in
full precision: perplexity on a fixed text, and how many greedy tokens stay the same.
"""

import argparse
import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional

from nanoengine.config import EngineSettings, load_engine_settings
from nanoengine.constants import QUANTIZATION_INT8, TORCH_DTYPES
from nanoengine.engine.generation.cached_generator import CachedGenerator
from nanoengine.engine.generation.model_runner import ModelRunner
from nanoengine.engine.interface import GenerationRequest, SamplingParams
from nanoengine.engine.memory.contiguous_allocator import ContiguousSlotAllocator
from nanoengine.engine.memory.kv_cache_pool import KVCachePool
from nanoengine.models.int8_quantization import quantize_linear_layers
from nanoengine.models.qwen3 import Qwen3ForCausalLM
from nanoengine.models.text_tokenizer import load_text_tokenizer
from nanoengine.models.weight_loader import download_model_files, load_qwen3_model

QUALITY_TEXT = (
    "Large language models generate text one token at a time. For every new token the model "
    "reads the whole conversation so far, which is why serving systems keep the keys and values "
    "of earlier tokens in a cache instead of computing them again. The cache grows with every "
    "token, so memory, not arithmetic, often decides how many people one graphics card can serve. "
    "Engineers therefore batch many requests together, hand out cache memory in small blocks, "
    "share the memory of identical prompt beginnings, and split long prompts into chunks so that "
    "short answers keep flowing. Some systems also let a small model guess several tokens that a "
    "large model then checks in a single pass, and many store weights and cache entries with "
    "fewer bits. Each of these ideas trades something: memory for speed, simplicity for "
    "throughput, or a little accuracy for a lot of capacity. Measuring that trade honestly, on "
    "the same hardware and the same workloads, is the only way to know whether it was worth it."
)
NUM_PROMPTS = 8
PROMPT_TOKENS = 24
GENERATED_TOKENS = 32
QUALITY_FILE_STEM = "quality-int8"


@dataclass(frozen=True)
class QualityReport:
    perplexity_full_precision: float
    perplexity_configured: float
    greedy_token_agreement: float
    identical_completions: int
    num_completions: int
    settings: str


def compute_perplexity(
    model: Qwen3ForCausalLM, token_ids: list[int], device: torch.device
) -> float:
    with torch.inference_mode():
        inputs = torch.tensor([token_ids], device=device)
        logits = model.compute_logits(model(inputs))[0, :-1].float()
        loss = functional.cross_entropy(logits, inputs[0, 1:])
    return math.exp(loss.item())


def build_cached_generator(
    model: Qwen3ForCausalLM, settings: EngineSettings, is_int8_cache: bool
) -> CachedGenerator:
    device = torch.device(settings.device)
    length = PROMPT_TOKENS + GENERATED_TOKENS
    pool = KVCachePool(model.config, length, device, TORCH_DTYPES[settings.dtype], is_int8_cache)
    runner = ModelRunner(model, pool, ContiguousSlotAllocator(1, length, device), device)
    return CachedGenerator(runner, frozenset(), device)


def generate_greedy(generator: CachedGenerator, prompt: list[int], request_id: str) -> list[int]:
    params = SamplingParams(max_new_tokens=GENERATED_TOKENS, ignore_eos=True)
    request = GenerationRequest(request_id, prompt, params)
    return [event.token_id for event in generator.generate(request)]


def evaluate_quality(
    full_model: Qwen3ForCausalLM,
    configured_model: Qwen3ForCausalLM,
    token_ids: list[int],
    settings: EngineSettings,
) -> QualityReport:
    device = torch.device(settings.device)
    full_generator = build_cached_generator(full_model, settings, is_int8_cache=False)
    configured_generator = build_cached_generator(
        configured_model, settings, settings.is_int8_kv_cache
    )
    matches, identical = 0, 0
    for index in range(NUM_PROMPTS):
        prompt = token_ids[index * PROMPT_TOKENS : (index + 1) * PROMPT_TOKENS]
        expected = generate_greedy(full_generator, prompt, f"full-{index}")
        actual = generate_greedy(configured_generator, prompt, f"configured-{index}")
        matches += sum(left == right for left, right in zip(expected, actual, strict=True))
        identical += expected == actual
    return QualityReport(
        perplexity_full_precision=compute_perplexity(full_model, token_ids, device),
        perplexity_configured=compute_perplexity(configured_model, token_ids, device),
        greedy_token_agreement=matches / (NUM_PROMPTS * GENERATED_TOKENS),
        identical_completions=identical,
        num_completions=NUM_PROMPTS,
        settings=f"quantization={settings.quantization}, kv_cache_dtype={settings.kv_cache_dtype}",
    )


def render_quality_markdown(report: QualityReport) -> str:
    return "\n".join(
        [
            f"# int8 quality ({report.settings})",
            "",
            "| Metric | Full precision | Configured |",
            "|---|---|---|",
            f"| Perplexity on the fixed text | {report.perplexity_full_precision:.3f} "
            f"| {report.perplexity_configured:.3f} |",
            f"| Greedy token agreement | 100% | {report.greedy_token_agreement:.1%} |",
            f"| Identical completions | {report.num_completions} "
            f"| {report.identical_completions} |",
            "",
        ]
    )


def main(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="nanoengine.benchmarks.quality")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parsed = parser.parse_args(arguments)
    settings = load_engine_settings()
    device = torch.device(settings.device)
    model_dir = download_model_files(settings.model_id, settings.model_cache_dir)
    full_model = load_qwen3_model(model_dir, device, TORCH_DTYPES[settings.dtype])
    configured_model = copy.deepcopy(full_model)
    if settings.quantization == QUANTIZATION_INT8:
        quantize_linear_layers(configured_model)
    token_ids = load_text_tokenizer(model_dir).encode(QUALITY_TEXT)
    report = evaluate_quality(full_model, configured_model, token_ids, settings)
    parsed.output_dir.mkdir(parents=True, exist_ok=True)
    stem = parsed.output_dir / QUALITY_FILE_STEM
    stem.with_suffix(".md").write_text(render_quality_markdown(report), encoding="utf-8")
    stem.with_suffix(".json").write_text(json.dumps(asdict(report), indent=2) + "\n")


if __name__ == "__main__":
    main()
