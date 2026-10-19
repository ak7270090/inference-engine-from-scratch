# Concurrency sweep: short_chat on http://localhost:8000

Generated 2026-10-19 13:01 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 68.9 | 70.2 | 46 | 60 | 14.2 | 14.6 | 1856 | 1894 |
| 4 | 64 | 0 | 173.8 | 44.6 | 103 | 106 | 22.4 | 23.1 | 2946 | 2953 |
| 16 | 64 | 0 | 361.5 | 22.9 | 117 | 126 | 43.3 | 52.9 | 5664 | 5680 |
| 64 | 64 | 0 | 617.0 | 9.9 | 324 | 524 | 101.4 | 138.8 | 13181 | 13273 |
