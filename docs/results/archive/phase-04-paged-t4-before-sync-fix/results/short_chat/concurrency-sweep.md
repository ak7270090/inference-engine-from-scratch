# Concurrency sweep: short_chat on http://localhost:8000

Generated 2026-10-18 05:48 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 23.8 | 23.9 | 44 | 50 | 41.6 | 48.5 | 5363 | 5454 |
| 4 | 64 | 0 | 93.4 | 23.5 | 47 | 90 | 42.3 | 51.0 | 5489 | 5563 |
| 16 | 64 | 0 | 360.2 | 22.8 | 129 | 140 | 43.4 | 54.6 | 5692 | 5710 |
| 64 | 64 | 0 | 990.1 | 16.2 | 441 | 446 | 61.5 | 78.8 | 8267 | 8272 |
