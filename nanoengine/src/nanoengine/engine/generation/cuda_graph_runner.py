"""CUDA graphs for decode steps: record one step's hundreds of kernel launches once, replay them.

A small model's decode step is mostly launch overhead: 28 layers × ~20 small kernels, each
launched from Python. A CUDA graph records the whole step on the GPU and replays it with one
launch. Graphs need fixed shapes and fixed memory, so each (batch size, context length) bucket
gets its own graph and static input buffers; a step is padded up to its bucket. Padded rows write
to the scratch slot and are ignored.

On a CPU (tests) the same padded path runs without capture.
"""

import logging
from dataclasses import dataclass

import torch

from nanoengine.engine.memory.kv_cache_pool import KVCachePool
from nanoengine.engine.memory.slot_allocator import SlotAllocator
from nanoengine.engine.sequence_state import Sequence
from nanoengine.models.attention_context import AttentionContext, build_attention_mask
from nanoengine.models.qwen3 import Qwen3ForCausalLM

logger = logging.getLogger(__name__)

BATCH_SIZE_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
SMALLEST_CONTEXT_BUCKET = 256
WARMUP_RUNS_BEFORE_CAPTURE = 2
# memory left free for prefill steps and eager decodes after all captures
GPU_MEMORY_RESERVE_BYTES = 2 * 1024**3
# per (sequence, context token, KV head, dim): gathered keys and values, plus slack
CAPTURE_BYTES_FACTOR = 4


def build_buckets(largest: int, candidates: tuple[int, ...]) -> list[int]:
    return sorted({value for value in candidates if value < largest} | {largest})


def build_context_buckets(max_model_len: int) -> list[int]:
    """Powers of two from 256, ending with ``max_model_len``."""
    buckets, length = [], SMALLEST_CONTEXT_BUCKET
    while length < max_model_len:
        buckets.append(length)
        length *= 2
    return [*buckets, max_model_len]


def round_up_to_bucket(value: int, buckets: list[int]) -> int:
    return next(bucket for bucket in buckets if bucket >= value)


@dataclass
class DecodeGraph:
    """Static inputs and output of one bucket; a CUDA graph reads and writes exactly these."""

    token_ids: torch.Tensor  # [b, 1]
    positions: torch.Tensor  # [b, 1]
    slot_mapping: torch.Tensor  # [b, 1]
    context_slots: torch.Tensor  # [b, c]
    context_lengths: torch.Tensor  # [b]
    logits: torch.Tensor | None = None  # [b, vocab], filled by capture or eager run
    graph: torch.cuda.CUDAGraph | None = None


class DecodeGraphRunner:
    def __init__(
        self,
        model: Qwen3ForCausalLM,
        pool: KVCachePool,
        allocator: SlotAllocator,
        device: torch.device,
        max_num_seqs: int,
        max_context_length: int,
    ) -> None:
        """Graphs cover up to ``max_num_seqs`` sequences and ``max_context_length`` tokens;
        anything larger decodes eagerly."""
        self._model = model
        self._pool = pool
        self._allocator = allocator
        self._device = device
        self._uses_cuda_graphs = device.type == "cuda"
        self.batch_buckets = build_buckets(max_num_seqs, BATCH_SIZE_BUCKETS)
        self.context_buckets = build_context_buckets(max_context_length)
        self._graphs: dict[tuple[int, int], DecodeGraph] = {}
        self._graph_memory_pool = torch.cuda.graph_pool_handle() if self._uses_cuda_graphs else None
        # one warm-up stream for every capture: the allocator caches freed memory per stream
        self._warmup_stream = torch.cuda.Stream(device) if self._uses_cuda_graphs else None

    @property
    def num_captured_graphs(self) -> int:
        return sum(1 for graph in self._graphs.values() if graph.graph is not None)

    @torch.inference_mode()
    def capture_all_buckets(self) -> None:
        """Capture every bucket up front, so no request waits for a capture while being served.

        Largest first: graphs share one memory pool, and smaller graphs can then reuse the
        memory the largest one needed. Under inference mode: with autograd on, each graph's
        output would keep every layer's activations alive for a backward pass that never comes.
        """
        for batch_size in reversed(self.batch_buckets):
            for context_length in reversed(self.context_buckets):
                if (batch_size, context_length) not in self._graphs:
                    self._create_graph(batch_size, context_length)

    def can_run(self, sequences: list[Sequence]) -> bool:
        """Decode steps (one uncached token each) that fit a bucket. On a GPU, a bucket whose
        capture failed or was skipped decodes eagerly: padding it would only add work."""
        fits = (
            len(sequences) <= self.batch_buckets[-1]
            and max(sequence.num_tokens for sequence in sequences) <= self.context_buckets[-1]
            and all(sequence.num_uncached_tokens == 1 for sequence in sequences)
        )
        if not fits or not self._uses_cuda_graphs:
            return fits
        bucket = self._bucket_for(sequences)
        return bucket not in self._graphs or self._graphs[bucket].graph is not None

    def _bucket_for(self, sequences: list[Sequence]) -> tuple[int, int]:
        batch_size = round_up_to_bucket(len(sequences), self.batch_buckets)
        context_length = round_up_to_bucket(
            max(sequence.num_tokens for sequence in sequences), self.context_buckets
        )
        return batch_size, context_length

    def run(self, sequences: list[Sequence]) -> torch.Tensor:
        """Logits ``[len(sequences), vocab]`` for each sequence's next token."""
        batch_size, context_length = self._bucket_for(sequences)
        decode_graph = self._graphs.get((batch_size, context_length))
        if decode_graph is None:
            decode_graph = self._create_graph(batch_size, context_length)
        self._fill_inputs(decode_graph, sequences)
        if decode_graph.graph is not None:
            decode_graph.graph.replay()
        else:
            decode_graph.logits = self._forward(decode_graph)
        return decode_graph.logits[: len(sequences)]

    def _fill_inputs(self, decode_graph: DecodeGraph, sequences: list[Sequence]) -> None:
        """Copy this step into the static buffers; rows past the batch become padding."""
        count = len(sequences)
        context_length = decode_graph.context_slots.shape[1]
        last_positions = [sequence.num_tokens - 1 for sequence in sequences]
        positions = torch.tensor(last_positions, device=self._device)[:, None]
        context_slots = self._allocator.build_context_slots(sequences, context_length)
        self._fill_padding(decode_graph)
        decode_graph.token_ids[:count, 0] = torch.tensor(
            [sequence.token_ids[-1] for sequence in sequences], device=self._device
        )
        decode_graph.positions[:count] = positions
        decode_graph.context_slots[:count] = context_slots
        decode_graph.slot_mapping[:count] = context_slots.gather(1, positions)
        decode_graph.context_lengths[:count] = torch.tensor(
            [sequence.num_tokens for sequence in sequences], device=self._device
        )

    def _fill_padding(self, decode_graph: DecodeGraph) -> None:
        """Padding rows: token 0 at position 0, writing and reading only the scratch slot."""
        decode_graph.token_ids.zero_()
        decode_graph.positions.zero_()
        decode_graph.slot_mapping.fill_(self._pool.scratch_slot)
        decode_graph.context_slots.fill_(self._pool.scratch_slot)
        decode_graph.context_lengths.fill_(1)

    def _forward(self, decode_graph: DecodeGraph) -> torch.Tensor:
        context_length = decode_graph.context_slots.shape[1]
        context = AttentionContext(
            positions=decode_graph.positions,
            slot_mapping=decode_graph.slot_mapping,
            context_slots=decode_graph.context_slots,
            attention_mask=build_attention_mask(
                decode_graph.positions, decode_graph.context_lengths, context_length
            ),
            key_caches=self._pool.key_caches,
            value_caches=self._pool.value_caches,
            key_scales=self._pool.key_scales,
            value_scales=self._pool.value_scales,
        )
        hidden_states = self._model(decode_graph.token_ids, context)
        return self._model.compute_logits(hidden_states[:, 0])

    def _create_graph(self, batch_size: int, context_length: int) -> DecodeGraph:
        """Allocate static buffers; on a GPU, warm up and capture the step into a CUDA graph."""
        index_options = {"device": self._device, "dtype": torch.long}
        decode_graph = DecodeGraph(
            token_ids=torch.zeros((batch_size, 1), **index_options),
            positions=torch.zeros((batch_size, 1), **index_options),
            slot_mapping=torch.zeros((batch_size, 1), **index_options),
            context_slots=torch.zeros((batch_size, context_length), **index_options),
            context_lengths=torch.ones(batch_size, **index_options),
        )
        self._fill_padding(decode_graph)
        if self._uses_cuda_graphs and not self._has_memory_for(batch_size, context_length):
            logger.warning(
                "skipping a decode graph to keep GPU memory free; this bucket runs eagerly",
                extra={"batch_size": batch_size, "context_length": context_length},
            )
        elif self._uses_cuda_graphs:
            try:
                self._capture(decode_graph)
            except torch.cuda.OutOfMemoryError:
                logger.warning(
                    "not enough GPU memory to capture a decode graph; this bucket runs eagerly",
                    extra={"batch_size": batch_size, "context_length": context_length},
                )
                decode_graph.graph = None
                torch.cuda.empty_cache()
        self._graphs[(batch_size, context_length)] = decode_graph
        return decode_graph

    def _has_memory_for(self, batch_size: int, context_length: int) -> bool:
        """Capture only if it leaves the reserve free: a failed capture can leak its memory."""
        free_bytes, _ = torch.cuda.mem_get_info(self._device)
        config = self._model.config
        element_size = self._model.model.embed_tokens.weight.element_size()
        needed_bytes = (
            CAPTURE_BYTES_FACTOR
            * batch_size
            * context_length
            * config.num_key_value_heads
            * config.head_dim
            * element_size
        )
        return free_bytes - needed_bytes > GPU_MEMORY_RESERVE_BYTES

    def _capture(self, decode_graph: DecodeGraph) -> None:
        torch.cuda.synchronize(self._device)
        self._warmup_stream.wait_stream(torch.cuda.current_stream(self._device))
        with torch.cuda.stream(self._warmup_stream):
            for _ in range(WARMUP_RUNS_BEFORE_CAPTURE):
                self._forward(decode_graph)
        torch.cuda.current_stream(self._device).wait_stream(self._warmup_stream)
        torch.cuda.synchronize(self._device)
        decode_graph.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(decode_graph.graph, pool=self._graph_memory_pool):
            decode_graph.logits = self._forward(decode_graph)
