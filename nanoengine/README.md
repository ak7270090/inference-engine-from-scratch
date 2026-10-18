# nanoengine

The inference engine: Qwen3 in plain PyTorch behind an OpenAI-style API. There is one package, and
each phase adds modules and a new value for a mode flag. Older values stay as baselines, so any
phase can be re-run and compared on the same code.

## Layout

```
nanoengine/
├── src/nanoengine/
│   ├── models/        qwen3.py, weight_loader.py, text_tokenizer.py
│   ├── engine/        interface.py, sampling.py, naive_generator.py, engine_loop.py,
│   │                  runtime.py (picks modules by mode), metrics.py
│   ├── api/           routes.py, schemas.py, completion_responses.py
│   ├── app.py, config.py, constants.py, step_timing.py, __main__.py
├── profiles/          engine modes per phase (phase-01-naive.env, ...)
├── harness/           harness expectations per phase and device (phase-01-naive-cpu.env, ...)
└── tests/
```

## Modes

| Setting (`NANOENGINE_*`) | Values so far | Added in |
|---|---|---|
| `KV_CACHE` | `none` | phase 1 |
| `SCHEDULER` | `serial` | phase 1 |

A profile in `profiles/` fixes the modes for one phase. Other settings: `MODEL_ID`, `DEVICE`,
`DTYPE`, `MAX_MODEL_LEN`, `MAX_WAITING_REQUESTS`, `PORT` (see `.env.example`).

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
