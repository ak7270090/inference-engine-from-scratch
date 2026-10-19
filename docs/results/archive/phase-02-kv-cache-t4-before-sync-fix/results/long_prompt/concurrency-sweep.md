# Concurrency sweep: long_prompt on http://localhost:8000

Generated 2026-10-18 06:06 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 0 | 19.7 | 20.8 | 130 | 175 | 47.9 | 55.3 | 1623 | 1664 |
| 4 | 16 | 0 | 20.0 | 20.9 | 4914 | 5026 | 47.6 | 53.2 | 6402 | 6503 |
