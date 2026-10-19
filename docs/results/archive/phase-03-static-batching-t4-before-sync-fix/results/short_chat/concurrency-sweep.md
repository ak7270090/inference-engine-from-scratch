# Concurrency sweep: short_chat on http://localhost:8000

Generated 2026-10-18 07:07 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 22.7 | 23.2 | 95 | 123 | 42.9 | 51.7 | 5576 | 6086 |
| 4 | 64 | 0 | 89.4 | 22.6 | 97 | 99 | 43.9 | 52.2 | 5717 | 5827 |
| 16 | 64 | 0 | 343.3 | 21.8 | 150 | 154 | 45.3 | 51.7 | 5972 | 6075 |
| 64 | 64 | 0 | 340.7 | 21.5 | 6281 | 18274 | 46.1 | 56.9 | 12193 | 24045 |
