# Concurrency sweep: short_chat on http://localhost:8000

Generated 2026-10-18 16:40 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 23.7 | 23.8 | 46 | 51 | 41.7 | 49.4 | 5389 | 5537 |
| 4 | 64 | 0 | 89.2 | 22.6 | 132 | 137 | 43.8 | 51.5 | 5750 | 5826 |
| 16 | 64 | 0 | 291.4 | 18.4 | 140 | 153 | 54.1 | 61.1 | 7025 | 7064 |
| 64 | 64 | 0 | 360.9 | 5.7 | 345 | 788 | 174.2 | 218.5 | 22544 | 22695 |
