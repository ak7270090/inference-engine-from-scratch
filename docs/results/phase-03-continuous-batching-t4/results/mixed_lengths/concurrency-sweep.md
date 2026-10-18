# Concurrency sweep: mixed_lengths on http://localhost:8000

Generated 2026-10-18 07:44 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 26.4 | 26.3 | 40 | 45 | 37.7 | 43.7 | 11051 | 18811 |
| 4 | 64 | 0 | 100.1 | 25.6 | 78 | 84 | 38.5 | 74.5 | 11462 | 19506 |
| 16 | 64 | 0 | 285.0 | 21.3 | 85 | 123 | 46.5 | 85.1 | 12527 | 25007 |
| 64 | 64 | 0 | 302.1 | 21.1 | 13018 | 37802 | 45.0 | 89.3 | 28998 | 57225 |
