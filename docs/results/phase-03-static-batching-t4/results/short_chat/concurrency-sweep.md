# Concurrency sweep: short_chat on http://localhost:8000

Generated 2026-10-18 08:18 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 27.5 | 27.8 | 88 | 103 | 35.7 | 41.5 | 4648 | 4736 |
| 4 | 64 | 0 | 107.8 | 27.3 | 89 | 93 | 36.4 | 42.9 | 4741 | 4850 |
| 16 | 64 | 0 | 416.4 | 26.5 | 140 | 143 | 37.4 | 42.6 | 4925 | 4949 |
| 64 | 64 | 0 | 419.1 | 26.4 | 5091 | 14828 | 37.5 | 42.2 | 9910 | 19544 |
