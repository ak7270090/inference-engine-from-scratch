# Concurrency sweep: long_prompt on http://localhost:8000

Generated 2026-10-18 11:54 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 0 | 23.6 | 23.7 | 46 | 52 | 42.0 | 48.2 | 1355 | 1405 |
| 4 | 16 | 0 | 70.4 | 18.6 | 147 | 152 | 53.9 | 73.0 | 1814 | 1834 |
| 16 | 16 | 0 | 90.1 | 5.8 | 369 | 372 | 171.8 | 176.7 | 5675 | 5678 |
| 64 | 16 | 0 | 89.7 | 5.8 | 372 | 374 | 171.7 | 175.1 | 5702 | 5704 |
