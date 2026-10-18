# Concurrency sweep: shared_prefix on http://localhost:8000

Generated 2026-10-18 09:44 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 27.2 | 27.4 | 50 | 66 | 36.2 | 43.3 | 2349 | 2438 |
| 4 | 64 | 0 | 27.7 | 27.9 | 6973 | 7135 | 35.6 | 42.0 | 9231 | 9412 |
