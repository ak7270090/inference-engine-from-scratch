"""Everything a running server needs: the engine loop, tokenizer, metrics and model limits."""

from dataclasses import dataclass

import torch

from nanoengine.config import EngineSettings
from nanoengine.constants import KV_CACHE_NONE, SCHEDULER_SERIAL, TORCH_DTYPES
from nanoengine.engine.engine_loop import EngineLoop
from nanoengine.engine.metrics import EngineMetrics
from nanoengine.engine.naive_generator import NaiveGenerator
from nanoengine.models.qwen3 import Qwen3ForCausalLM
from nanoengine.models.text_tokenizer import TextTokenizer, load_text_tokenizer
from nanoengine.models.weight_loader import download_model_files, load_qwen3_model

GENERATORS_BY_KV_CACHE = {KV_CACHE_NONE: NaiveGenerator}
ENGINE_LOOPS_BY_SCHEDULER = {SCHEDULER_SERIAL: EngineLoop}


@dataclass
class EngineRuntime:
    engine: EngineLoop
    tokenizer: TextTokenizer
    metrics: EngineMetrics
    served_model_name: str
    max_model_len: int


def build_engine_runtime(
    model: Qwen3ForCausalLM, tokenizer: TextTokenizer, settings: EngineSettings
) -> EngineRuntime:
    """Pick the generator and request loop for the configured ``kv_cache`` and ``scheduler``."""
    metrics = EngineMetrics()
    generator_class = GENERATORS_BY_KV_CACHE[settings.kv_cache]
    engine_loop_class = ENGINE_LOOPS_BY_SCHEDULER[settings.scheduler]
    generator = generator_class(model, tokenizer.stop_token_ids, torch.device(settings.device))
    return EngineRuntime(
        engine=engine_loop_class(generator, metrics, settings.max_waiting_requests),
        tokenizer=tokenizer,
        metrics=metrics,
        served_model_name=settings.public_model_name,
        max_model_len=settings.max_model_len,
    )


def create_engine_runtime(settings: EngineSettings) -> EngineRuntime:
    """Download (or reuse) the checkpoint, load it onto the configured device, build the engine."""
    model_dir = download_model_files(settings.model_id, settings.model_cache_dir)
    model = load_qwen3_model(model_dir, torch.device(settings.device), TORCH_DTYPES[settings.dtype])
    return build_engine_runtime(model, load_text_tokenizer(model_dir), settings)
