# Concurrency sweep: long_prompt on http://localhost:8000

Generated 2026-10-18 08:22 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 0 | 24.7 | 26.3 | 111 | 154 | 37.7 | 44.3 | 1288 | 1355 |
| 4 | 16 | 0 | 77.4 | 25.4 | 449 | 451 | 38.6 | 154.7 | 1655 | 1671 |
| 16 | 16 | 0 | 111.3 | 9.2 | 1241 | 1784 | 90.1 | 1101.5 | 4599 | 4601 |
| 64 | 16 | 0 | 110.6 | 8.6 | 1019 | 1805 | 90.7 | 903.8 | 4624 | 4627 |
