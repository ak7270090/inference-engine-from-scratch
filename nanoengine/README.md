# nanoengine

The inference engine: Qwen3 in plain PyTorch behind an OpenAI-style API. There is one package, and
each phase adds modules and a new value for a mode flag. Older values stay as baselines, so any
phase can be re-run and compared on the same code.

## Layout

```
nanoengine/
├── src/nanoengine/
│   ├── models/        qwen3.py, attention_context.py (cached attention over slot pools),
│   │                  weight_loader.py, text_tokenizer.py
│   ├── engine/
│   │   ├── generation/   naive_generator, cached_generator, model_runner, batch_builder
│   │   ├── memory/       kv_cache_pool, contiguous_allocator, block_allocator, paged_allocator
│   │   ├── scheduling/   request_queue, serial_engine_loop, batch_engine_loop, batch_scheduler
│   │   └── interface, sampling, sequence_state, metrics, runtime (builds the engine per mode)
│   ├── api/           routes.py, schemas.py, completion_responses.py
│   ├── app.py, config.py, constants.py, step_timing.py, __main__.py
├── profiles/          engine modes per phase (phase-01-naive.env, ...)
├── harness/           harness expectations per phase and device (phase-01-naive-cpu.env, ...)
└── tests/
```

## Modes

| Setting (`NANOENGINE_*`) | Values so far | Added in |
|---|---|---|
| `KV_CACHE` | `none` (recompute everything), `contiguous` (a `max_model_len` region per sequence), `paged` (16-token blocks on demand) | 1, 2, 4 |
| `SCHEDULER` | `serial` (one request at a time), `static` (fill a batch, run it to the end), `continuous` (join and leave every step) | 1, 3, 3 |
| `KV_CACHE_MEMORY_MB`, `KV_BLOCK_SIZE` | KV pool budget (default 1024 MB) and block size (16) | 3, 4 |
| `MAX_NUM_SEQS`, `MAX_PREFILL_TOKENS`, `BATCH_FILL_TIMEOUT_MS` | seats per step (64), prompt tokens per prefill step (4096), static fill wait (50 ms) | 3 |

A profile in `profiles/` fixes the modes for one phase. Batching needs a KV cache
(`scheduler=static|continuous` with `kv_cache=none` is rejected). Every mode is checked to give the
same greedy tokens as recomputing each request alone.

How one step works with a KV cache: the KV cache is a flat pool of token slots per layer. A
contiguous cache gives each sequence a fixed range of slots; a paged cache gives it blocks of slots
anywhere in the pool (its block table maps positions to slots). Attention writes the new tokens'
keys and values into their slots, then reads each sequence's context through its slot list.

Other settings: `MODEL_ID`, `DEVICE`, `DTYPE`, `MAX_MODEL_LEN`, `MAX_WAITING_REQUESTS`, `PORT`
(see `.env.example`).

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest                     # unit tests, 80% coverage
$env:NANOENGINE_ENV_FILE="profiles/phase-01-naive.env"
.\.venv\Scripts\python.exe -m nanoengine                 # serves on :8000
```

The API docs page is at `http://localhost:8000/docs`:

![The four endpoints on the engine's API docs page](../docs/images/phase-01/api-docs.png)

Set `NANOENGINE_TEST_REAL_WEIGHTS=1` to run the logits check against the real Qwen3-0.6B weights
(downloaded once, about 1.2 GB).

## Phases

Each phase's write-up, results and screenshots are in [docs/phases/](../docs/phases/). The code at
the end of each phase has a git tag (`phase-01`, `phase-02`, ...).
