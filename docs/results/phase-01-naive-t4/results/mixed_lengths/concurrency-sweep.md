# Concurrency sweep: mixed_lengths on http://localhost:8000

Generated 2026-10-17 20:40 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 27.3 | 28.5 | 36 | 45 | 35.0 | 54.0 | 10001 | 19846 |
| 4 | 64 | 0 | 27.3 | 28.3 | 29555 | 46691 | 35.0 | 53.8 | 40051 | 59260 |
