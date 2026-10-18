# Concurrency sweep: shared_prefix on http://localhost:8000

Generated 2026-10-18 06:42 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 21.8 | 21.9 | 62 | 79 | 45.4 | 52.9 | 2936 | 3016 |
| 4 | 64 | 0 | 80.7 | 21.5 | 246 | 259 | 46.1 | 54.0 | 3165 | 3246 |
| 16 | 64 | 0 | 211.2 | 14.9 | 648 | 949 | 61.9 | 132.6 | 4846 | 4870 |
| 64 | 64 | 0 | 208.0 | 13.9 | 6305 | 16696 | 67.5 | 251.0 | 10723 | 19684 |
