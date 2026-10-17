# Concurrency sweep: shared_prefix on http://localhost:8000

Generated 2026-10-17 19:10 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 117.7 | 121.7 | 24 | 35 | 8.2 | 9.2 | 542 | 585 |
| 4 | 64 | 0 | 300.7 | 78.4 | 48 | 61 | 12.8 | 14.0 | 851 | 859 |
| 16 | 64 | 0 | 503.0 | 32.3 | 87 | 98 | 31.0 | 33.9 | 2034 | 2046 |
| 64 | 64 | 0 | 641.5 | 10.4 | 327 | 333 | 96.1 | 102.3 | 6372 | 6379 |
