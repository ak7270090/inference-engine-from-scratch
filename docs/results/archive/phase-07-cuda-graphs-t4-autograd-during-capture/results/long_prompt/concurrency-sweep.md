# Concurrency sweep: long_prompt on http://localhost:8000

Generated 2026-10-18 18:27 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 0 | 24.3 | 24.5 | 45 | 52 | 40.6 | 48.7 | 1308 | 1369 |
| 4 | 16 | 0 | 71.9 | 19.0 | 145 | 146 | 52.9 | 57.0 | 1778 | 1784 |
| 16 | 16 | 16 | 0.0 | 0.0 | 0 | 0 | 0.0 | 0.0 | 0 | 0 |
| 64 | 16 | 16 | 0.0 | 0.0 | 0 | 0 | 0.0 | 0.0 | 0 | 0 |
