# Concurrency sweep: long_prompt on http://localhost:8000

Generated 2026-10-18 13:28 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 0 | 17.9 | 18.0 | 60 | 88 | 55.3 | 61.4 | 1787 | 1814 |
| 4 | 16 | 0 | 51.4 | 13.6 | 194 | 227 | 74.0 | 75.6 | 2480 | 2512 |
| 16 | 16 | 0 | 71.6 | 4.6 | 430 | 432 | 217.3 | 239.2 | 7145 | 7147 |
| 64 | 16 | 0 | 71.6 | 4.6 | 419 | 422 | 217.4 | 221.1 | 7144 | 7147 |
