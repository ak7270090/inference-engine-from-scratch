# Concurrency sweep: long_prompt on http://localhost:8000

Generated 2026-10-18 16:40 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 0 | 23.2 | 23.4 | 47 | 54 | 42.5 | 48.3 | 1377 | 1432 |
| 4 | 16 | 0 | 70.6 | 18.6 | 149 | 155 | 53.6 | 67.5 | 1809 | 1825 |
| 16 | 16 | 0 | 90.8 | 5.9 | 360 | 363 | 171.5 | 173.8 | 5632 | 5635 |
| 64 | 16 | 0 | 91.2 | 5.9 | 352 | 354 | 170.4 | 173.1 | 5608 | 5610 |
