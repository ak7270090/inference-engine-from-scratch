# Concurrency sweep: shared_prefix on http://localhost:8000

Generated 2026-10-18 08:23 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 27.1 | 27.9 | 100 | 119 | 35.7 | 41.8 | 2361 | 2426 |
| 4 | 64 | 0 | 98.6 | 27.1 | 272 | 278 | 36.6 | 42.7 | 2599 | 2638 |
| 16 | 64 | 0 | 154.0 | 27.0 | 3064 | 5831 | 36.8 | 42.2 | 5406 | 8167 |
| 64 | 64 | 0 | 154.3 | 27.1 | 11093 | 22758 | 36.7 | 40.5 | 13420 | 25072 |
