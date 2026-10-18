# Concurrency sweep: long_prompt on http://localhost:8000

Generated 2026-10-18 07:08 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 0 | 21.0 | 23.0 | 164 | 213 | 43.3 | 48.4 | 1518 | 1561 |
| 4 | 16 | 0 | 67.4 | 22.7 | 534 | 542 | 43.7 | 51.8 | 1902 | 1927 |
| 16 | 16 | 0 | 68.0 | 22.8 | 3311 | 6152 | 43.7 | 57.0 | 4681 | 7529 |
| 64 | 16 | 0 | 66.3 | 22.5 | 3436 | 6307 | 44.4 | 52.7 | 4811 | 7723 |
