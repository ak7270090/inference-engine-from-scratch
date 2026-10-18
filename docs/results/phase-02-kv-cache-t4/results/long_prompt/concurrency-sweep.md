# Concurrency sweep: long_prompt on http://localhost:8000

Generated 2026-10-18 09:39 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 0 | 25.7 | 27.5 | 113 | 162 | 36.1 | 43.1 | 1239 | 1327 |
| 4 | 16 | 0 | 25.6 | 27.5 | 3831 | 4008 | 36.3 | 43.6 | 4963 | 5131 |
