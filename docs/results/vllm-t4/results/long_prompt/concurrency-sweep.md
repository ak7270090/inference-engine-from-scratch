# Concurrency sweep: long_prompt on http://localhost:8000

Generated 2026-10-17 19:08 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 0 | 95.6 | 104.0 | 29 | 50 | 9.7 | 16.4 | 328 | 422 |
| 4 | 16 | 0 | 210.0 | 56.5 | 60 | 66 | 17.8 | 19.0 | 608 | 613 |
| 16 | 16 | 0 | 298.2 | 19.7 | 142 | 145 | 50.9 | 52.2 | 1713 | 1716 |
| 64 | 16 | 0 | 295.7 | 19.8 | 158 | 160 | 51.0 | 52.6 | 1727 | 1729 |
