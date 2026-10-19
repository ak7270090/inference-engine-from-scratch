# Concurrency sweep: shared_prefix on http://localhost:8000

Generated 2026-10-19 13:05 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 44.7 | 45.5 | 46 | 61 | 22.0 | 22.3 | 1433 | 1449 |
| 4 | 64 | 0 | 77.7 | 19.8 | 111 | 114 | 50.4 | 52.4 | 3293 | 3298 |
| 16 | 64 | 0 | 158.8 | 10.1 | 230 | 238 | 98.9 | 105.5 | 6443 | 6459 |
| 64 | 64 | 0 | 174.4 | 2.8 | 853 | 1781 | 354.5 | 528.9 | 23258 | 23474 |
