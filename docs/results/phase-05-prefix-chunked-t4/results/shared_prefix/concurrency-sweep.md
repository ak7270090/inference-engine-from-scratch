# Concurrency sweep: shared_prefix on http://localhost:8000

Generated 2026-10-18 11:59 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 23.7 | 23.7 | 46 | 50 | 41.9 | 47.8 | 2703 | 2764 |
| 4 | 64 | 0 | 89.0 | 22.9 | 133 | 142 | 43.2 | 55.7 | 2858 | 2954 |
| 16 | 64 | 0 | 156.5 | 10.0 | 228 | 236 | 100.4 | 107.0 | 6540 | 6549 |
| 64 | 64 | 0 | 170.7 | 2.8 | 595 | 1715 | 363.7 | 506.4 | 23637 | 23985 |
