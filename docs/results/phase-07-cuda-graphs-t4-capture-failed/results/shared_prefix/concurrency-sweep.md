# Concurrency sweep: shared_prefix on http://localhost:8000

Generated 2026-10-18 16:45 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 23.0 | 23.1 | 47 | 53 | 42.9 | 52.9 | 2777 | 2966 |
| 4 | 64 | 0 | 75.2 | 19.3 | 136 | 159 | 51.7 | 57.4 | 3401 | 3439 |
| 16 | 64 | 0 | 97.1 | 6.1 | 235 | 240 | 163.4 | 172.9 | 10542 | 10559 |
| 64 | 64 | 60 | 25.7 | 19.1 | 6643 | 6647 | 51.5 | 65.2 | 9944 | 9947 |
