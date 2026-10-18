# Concurrency sweep: long_prompt on http://localhost:8000

Generated 2026-10-18 06:37 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 0 | 20.8 | 22.0 | 118 | 173 | 45.5 | 53.4 | 1529 | 1647 |
| 4 | 16 | 0 | 65.6 | 21.3 | 491 | 505 | 46.6 | 81.5 | 1950 | 2016 |
| 16 | 16 | 0 | 99.4 | 8.1 | 1296 | 2080 | 98.4 | 1150.8 | 5144 | 5148 |
| 64 | 16 | 0 | 98.1 | 8.0 | 1331 | 2130 | 99.7 | 1434.4 | 5215 | 5216 |
