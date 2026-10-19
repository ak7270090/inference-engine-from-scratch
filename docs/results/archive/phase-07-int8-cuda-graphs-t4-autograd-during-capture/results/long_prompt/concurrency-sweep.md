# Concurrency sweep: long_prompt on http://localhost:8000

Generated 2026-10-18 18:37 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 0 | 17.4 | 17.5 | 62 | 73 | 57.1 | 61.6 | 1840 | 1878 |
| 4 | 16 | 0 | 51.8 | 13.7 | 194 | 230 | 73.4 | 77.0 | 2460 | 2501 |
| 16 | 16 | 16 | 0.0 | 0.0 | 0 | 0 | 0.0 | 0.0 | 0 | 0 |
| 64 | 16 | 16 | 0.0 | 0.0 | 0 | 0 | 0.0 | 0.0 | 0 | 0 |
