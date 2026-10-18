"""Build the engine for the configured modes, plus what the API needs around it."""

from dataclasses import dataclass

import torch

from nanoengine.config import EngineSettings
from nanoengine.constants import (
    KV_CACHE_CONTIGUOUS,
    KV_CACHE_NONE,
    QUANTIZATION_INT8,
    SCHEDULER_STATIC,
    TORCH_DTYPES,
)
from nanoengine.engine.generation.cached_generator import CachedGenerator
from nanoengine.engine.generation.cuda_graph_runner import DecodeGraphRunner
from nanoengine.engine.generation.model_runner import ModelRunner
from nanoengine.engine.generation.naive_generator import NaiveGenerator
from nanoengine.engine.interface import InferenceEngine
from nanoengine.engine.memory.contiguous_allocator import ContiguousSlotAllocator
from nanoengine.engine.memory.kv_cache_pool import KVCachePool, count_slots_for_budget
from nanoengine.engine.memory.paged_allocator import PagedSlotAllocator
from nanoengine.engine.memory.prefix_cache import PrefixCachingSlotAllocator
from nanoengine.engine.memory.slot_allocator import SlotAllocator
from nanoengine.engine.metrics import EngineMetrics
from nanoengine.engine.scheduling.batch_engine_loop import BatchEngineLoop
from nanoengine.engine.scheduling.batch_scheduler import (
    ChunkedPrefillScheduler,
    ContinuousBatchScheduler,
    StaticBatchScheduler,
)
from nanoengine.engine.scheduling.serial_engine_loop import SerialEngineLoop
from nanoengine.models.int8_quantization import quantize_linear_layers
from nanoengine.models.qwen3 import Qwen3ForCausalLM
from nanoengine.models.text_tokenizer import TextTokenizer, load_text_tokenizer
from nanoengine.models.weight_loader import download_model_files, load_qwen3_model

MILLISECONDS_PER_SECOND = 1000


@dataclass
class EngineRuntime:
    engine: InferenceEngine
    tokenizer: TextTokenizer
    metrics: EngineMetrics
    served_model_name: str
    max_model_len: int


def create_slot_allocator(
    model: Qwen3ForCausalLM, settings: EngineSettings
) -> tuple[SlotAllocator, int]:
    """The allocator for ``kv_cache`` and the pool size in slots, from ``kv_cache_memory_mb``.

    Serial contiguous mode needs one ``max_model_len`` region; batched contiguous mode fits as
    many regions as the budget allows; paged mode splits the budget into blocks.
    """
    device = torch.device(settings.device)
    if settings.kv_cache == KV_CACHE_CONTIGUOUS:
        if settings.is_batched:
            budget_slots = count_slots_for_budget(
                model.config,
                TORCH_DTYPES[settings.dtype],
                settings.kv_cache_memory_mb,
                settings.is_int8_kv_cache,
            )
            num_regions = min(settings.max_num_seqs, budget_slots // settings.max_model_len)
        else:
            num_regions = 1
        if num_regions < 1:
            raise ValueError(
                f"kv_cache_memory_mb={settings.kv_cache_memory_mb} cannot hold one "
                f"max_model_len={settings.max_model_len} region; raise the budget"
            )
        allocator = ContiguousSlotAllocator(num_regions, settings.max_model_len, device)
        return allocator, num_regions * settings.max_model_len
    budget_slots = count_slots_for_budget(
        model.config,
        TORCH_DTYPES[settings.dtype],
        settings.kv_cache_memory_mb,
        settings.is_int8_kv_cache,
    )
    num_blocks = budget_slots // settings.kv_block_size
    if num_blocks * settings.kv_block_size < settings.max_model_len:
        raise ValueError(
            f"kv_cache_memory_mb={settings.kv_cache_memory_mb} holds fewer tokens than "
            f"max_model_len={settings.max_model_len}; raise the budget"
        )
    allocator_class = PrefixCachingSlotAllocator if settings.prefix_cache else PagedSlotAllocator
    allocator = allocator_class(num_blocks, settings.kv_block_size, device)
    return allocator, num_blocks * settings.kv_block_size


def create_batch_scheduler(
    allocator: SlotAllocator, settings: EngineSettings
) -> ContinuousBatchScheduler:
    if settings.scheduler == SCHEDULER_STATIC:
        return StaticBatchScheduler(
            allocator,
            settings.max_num_seqs,
            settings.max_prefill_tokens,
            settings.batch_fill_timeout_ms / MILLISECONDS_PER_SECOND,
        )
    if settings.chunked_prefill:
        return ChunkedPrefillScheduler(
            allocator, settings.max_num_seqs, settings.max_num_batched_tokens
        )
    return ContinuousBatchScheduler(allocator, settings.max_num_seqs, settings.max_prefill_tokens)


def create_engine(
    model: Qwen3ForCausalLM,
    stop_token_ids: frozenset[int],
    settings: EngineSettings,
    metrics: EngineMetrics,
) -> InferenceEngine:
    device = torch.device(settings.device)
    if settings.quantization == QUANTIZATION_INT8:
        quantize_linear_layers(model)
    if settings.kv_cache == KV_CACHE_NONE:
        generator = NaiveGenerator(model, stop_token_ids, device)
        return SerialEngineLoop(generator, metrics, settings.max_waiting_requests)
    allocator, num_slots = create_slot_allocator(model, settings)
    pool = KVCachePool(
        model.config, num_slots, device, TORCH_DTYPES[settings.dtype], settings.is_int8_kv_cache
    )
    decode_graphs = None
    if settings.cuda_graphs:
        decode_graphs = DecodeGraphRunner(
            model, pool, allocator, device, settings.max_num_seqs, settings.max_model_len
        )
        if device.type == "cuda":
            decode_graphs.capture_all_buckets()
    runner = ModelRunner(model, pool, allocator, device, decode_graphs)
    if not settings.is_batched:
        generator = CachedGenerator(runner, stop_token_ids, device)
        return SerialEngineLoop(generator, metrics, settings.max_waiting_requests)
    return BatchEngineLoop(
        runner,
        create_batch_scheduler(allocator, settings),
        metrics,
        stop_token_ids,
        settings.max_waiting_requests,
        device,
    )


def build_engine_runtime(
    model: Qwen3ForCausalLM, tokenizer: TextTokenizer, settings: EngineSettings
) -> EngineRuntime:
    metrics = EngineMetrics()
    return EngineRuntime(
        engine=create_engine(model, tokenizer.stop_token_ids, settings, metrics),
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
