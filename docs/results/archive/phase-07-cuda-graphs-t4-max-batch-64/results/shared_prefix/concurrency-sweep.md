# Concurrency sweep: shared_prefix on http://localhost:8000

Generated 2026-10-19 12:37 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 43.8 | 44.6 | 48 | 59 | 22.4 | 22.8 | 1461 | 1473 |
| 4 | 64 | 0 | 76.7 | 19.6 | 117 | 122 | 51.1 | 53.3 | 3338 | 3348 |
| 16 | 64 | 0 | 97.9 | 6.2 | 244 | 250 | 161.9 | 169.0 | 10457 | 10468 |
| 64 | 64 | 0 | 102.8 | 1.6 | 1011 | 2646 | 607.9 | 744.9 | 39276 | 39836 |
