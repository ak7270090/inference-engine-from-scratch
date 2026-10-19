# Concurrency sweep: long_prompt on http://localhost:8000

Generated 2026-10-19 12:33 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 0 | 22.6 | 22.7 | 48 | 53 | 43.8 | 54.0 | 1414 | 1474 |
| 4 | 16 | 0 | 70.8 | 18.7 | 149 | 154 | 53.7 | 55.2 | 1808 | 1812 |
| 16 | 16 | 0 | 91.3 | 5.9 | 360 | 363 | 170.5 | 173.5 | 5605 | 5607 |
| 64 | 16 | 0 | 91.4 | 5.9 | 349 | 351 | 170.0 | 174.3 | 5599 | 5601 |
