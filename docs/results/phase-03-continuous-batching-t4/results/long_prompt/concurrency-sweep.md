# Concurrency sweep: long_prompt on http://localhost:8000

Generated 2026-10-18 07:24 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 0 | 24.5 | 26.0 | 108 | 162 | 38.1 | 44.9 | 1301 | 1366 |
| 4 | 16 | 0 | 77.5 | 25.6 | 442 | 445 | 38.8 | 154.6 | 1652 | 1658 |
| 16 | 16 | 0 | 112.2 | 9.3 | 1229 | 1770 | 89.1 | 1091.5 | 4560 | 4562 |
| 64 | 16 | 0 | 111.7 | 9.0 | 1124 | 1791 | 89.7 | 984.4 | 4578 | 4580 |
