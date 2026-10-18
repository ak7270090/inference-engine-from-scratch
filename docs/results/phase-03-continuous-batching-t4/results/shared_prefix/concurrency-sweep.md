# Concurrency sweep: shared_prefix on http://localhost:8000

Generated 2026-10-18 07:28 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 26.1 | 26.2 | 50 | 67 | 37.8 | 43.8 | 2453 | 2513 |
| 4 | 64 | 0 | 96.3 | 25.7 | 218 | 223 | 38.4 | 48.4 | 2657 | 2730 |
| 16 | 64 | 0 | 236.7 | 16.8 | 583 | 855 | 55.0 | 116.2 | 4327 | 4337 |
| 64 | 64 | 0 | 235.2 | 15.5 | 5624 | 14919 | 60.5 | 208.1 | 9590 | 17408 |
