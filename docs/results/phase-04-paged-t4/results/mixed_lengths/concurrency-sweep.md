# Concurrency sweep: mixed_lengths on http://localhost:8000

Generated 2026-10-18 08:41 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 26.8 | 26.8 | 39 | 45 | 37.1 | 43.1 | 10875 | 18583 |
| 4 | 64 | 0 | 100.4 | 25.7 | 77 | 82 | 38.3 | 73.8 | 11420 | 19390 |
| 16 | 64 | 0 | 286.5 | 21.6 | 84 | 127 | 45.7 | 84.3 | 12497 | 24724 |
| 64 | 64 | 0 | 593.6 | 17.1 | 449 | 452 | 61.4 | 71.2 | 18355 | 28787 |
