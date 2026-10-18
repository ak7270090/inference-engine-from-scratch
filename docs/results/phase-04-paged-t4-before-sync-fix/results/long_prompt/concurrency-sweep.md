# Concurrency sweep: long_prompt on http://localhost:8000

Generated 2026-10-18 05:48 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 0 | 22.6 | 24.0 | 112 | 170 | 41.5 | 53.8 | 1403 | 1503 |
| 4 | 16 | 0 | 72.0 | 23.4 | 447 | 457 | 42.6 | 160.5 | 1779 | 1789 |
| 16 | 16 | 0 | 110.7 | 8.6 | 1012 | 1784 | 91.3 | 891.0 | 4622 | 4624 |
| 64 | 16 | 0 | 110.7 | 8.6 | 1008 | 1781 | 91.5 | 892.0 | 4624 | 4625 |
