# Concurrency sweep: mixed_lengths on http://localhost:8000

Generated 2026-10-19 14:58 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 26.1 | 26.5 | 67 | 71 | 36.9 | 45.7 | 11111 | 19573 |
| 4 | 64 | 0 | 66.8 | 17.0 | 181 | 232 | 58.7 | 108.6 | 16179 | 31100 |
| 16 | 64 | 0 | 125.6 | 8.6 | 273 | 335 | 121.0 | 191.1 | 25918 | 64294 |
| 64 | 64 | 0 | 241.2 | 6.8 | 379 | 833 | 152.9 | 178.2 | 44960 | 71460 |
