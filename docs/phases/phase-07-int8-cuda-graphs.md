# Phase 7: CUDA graphs, int8 weights and an int8 KV cache

**Profiles:**
- `phase-07-cuda-graphs`: phase 5 plus `cuda_graphs=true`.
- `phase-07-int8-cuda-graphs`: the same plus `quantization=int8` and `kv_cache_dtype=int8`.

**Code at the end of this phase:** `git checkout phase-07`.

![Phase 7](../images/plan-phase-7-quantization.png)

## What changed

| Module | Job |
|---|---|
| `engine/generation/cuda_graph_runner.py` | Records a whole decode step as a CUDA graph and replays it with one launch, instead of about 500 small kernel launches from Python. Graphs need fixed shapes, so steps are padded to (batch size, context length) buckets. Padded rows write to the scratch slot. Buckets are captured at startup under inference mode |
| `models/int8_quantization.py` | `Int8Linear` stores every projection's weight as int8 with one scale per output row. `quantize_vectors` stores KV entries as int8 with one scale per (token, head) |
| `models/attention_context.py` | Writes and reads the int8 KV cache. Grouped-query attention uses `enable_gqa` instead of copying each KV head per query head |
| `benchmarks/quality.py` | Perplexity on a fixed text and greedy-token agreement, int8 against fp16 |

## Results on a Kaggle T4

**CUDA graphs are the biggest single speed-up of the project.** A decode step with 128 tokens of
context drops from 33 ms to 13.3 ms. The plan asked for at least 30% less; this is 60%.

![Step time against sequence length: naive, KV cache, CUDA graphs](../images/progress-step-time.png)

| short_chat on a T4 | Phase 5 | **Phase 7a: CUDA graphs** | Phase 7b: + int8 | vLLM |
|---|---|---|---|---|
| 1 user: tok/s | 23.4 | **70.2** | 27.3 | 142.6 |
| 4 users: tok/s | 91 | **174** | 85 | 478 |
| 64 users: tok/s | 602 | **617** | 433 | 1,928 |
| One decode step, 128 tokens | 33 ms | **13.3 ms** | 33 ms | — |

One user now gets half of vLLM's speed; before graphs it got a sixth.

**Graphs pay off only for small batches.** A run with graphs for up to 64 sequences
([archived](../results/archive/phase-07-cuda-graphs-t4-max-batch-64/results/harness-report.md))
served 64 users at 363 tok/s, slower than without graphs. With graphs only up to 8 sequences (the
default, `cuda_graph_max_batch=8`), small batches get the speed-up and large batches keep eager
throughput. Small batches are where launch overhead dominates. In large batches the math does.

**int8: the quality cost is negligible, and plain PyTorch makes it slower.**

| int8 on a T4 | fp16 | int8 weights + int8 KV |
|---|---|---|
| Perplexity on a fixed text | 66.676 | 66.699 (+0.03%) |
| KV tokens in the same 4 GB budget | 37,440 | **73,744 (1.97x)** |
| One user, tok/s | 70.2 | 27.3 |

- **Memory:** the int8 KV cache stores 1 byte per number plus one fp16 scale per head, so the
  same budget holds 1.97x the tokens. The plan's criterion of 2x is met within the scale overhead.
- **Speed:** each matmul turns the int8 weight back into fp16 first, so int8 costs time here.
  Fast int8 inference needs kernels that multiply int8 directly.
- **Greedy agreement is a misleading measure.** It is 14.8%, but it compares tokens position by
  position, so one early different token makes the rest of the completion differ. Perplexity is
  the better measure of the quality cost here.

![Harness suites against phase 7a on a T4](../images/phase-07/terminal-phase-07-cuda-graphs-harness-t4.png)

## Five failed runs, and what each one taught

The archived runs record how CUDA graphs went from crashing to working:

1. **[Out of memory at capture](../results/archive/phase-07-cuda-graphs-t4-capture-oom/server.log).**
   Every capture had its own warm-up stream. The allocator caches freed memory per stream, so no
   capture reused another's memory. Fix: one warm-up stream, largest bucket first.
2. **[Graphs filled the GPU](../results/archive/phase-07-cuda-graphs-t4-graphs-filled-memory/server.log).**
   Failed captures leaked memory. Fixes: a free-memory check before each capture, graphs only up to
   1,024 tokens of context, and `enable_gqa` instead of per-head copies.
3. **[Every capture skipped or failed](../results/archive/phase-07-cuda-graphs-t4-capture-failed/server.log).**
   The benchmarks also ran out of memory, because they shared GPU 0 with the server. Fix: the
   benchmarks run on the second T4.
4. **[The root cause](../results/archive/phase-07-cuda-graphs-t4-autograd-during-capture/server.log).**
   Capture ran at startup outside `torch.inference_mode()` with trainable weights. Autograd saved
   every layer's activations, and each graph output kept them alive. The same run showed int8
   producing NaN, because `Int8Linear` multiplied by raw int8 values (up to 127) and scaled
   afterwards, which overflows fp16. Fixes: capture under inference mode, load weights with
   `requires_grad` off, and dequantize int8 weights before the matmul.
5. **[Graphs up to batch 64](../results/archive/phase-07-cuda-graphs-t4-max-batch-64/results/harness-report.md).**
   They worked, but large batches got slower. Fix: graphs only up to batch 8.

## Correctness

- **Padded graph path:** on CPU, the same bucket-padded path gives exactly the tokens of the eager
  path.
- **Failure handling:** a failed or skipped capture falls back to eager decoding. A tested graph
  output carries no autograd state.
- **int8:** `Int8Linear` stays within 2% of fp16 and does not overflow in fp16. A quantized model's
  logits have cosine similarity above 0.99 with the original's.
- **Real Qwen3-0.6B on CPU:** the CUDA-graph path gives 10/10 identical outputs.

Full results: [7a T4](../results/phase-07-cuda-graphs-t4/results/harness-report.md) ·
[7b T4](../results/phase-07-int8-cuda-graphs-t4/results/harness-report.md) ·
[int8 quality](../results/phase-07-int8-cuda-graphs-t4/results/quality-int8.md) ·
[CPU](../results/phase-07-cuda-graphs-cpu/harness-report.md).
