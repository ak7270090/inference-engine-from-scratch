# Concurrency sweep: shared_prefix on http://localhost:8000

Generated 2026-10-18 13:34 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 17.8 | 17.9 | 59 | 64 | 55.5 | 64.5 | 3587 | 3689 |
| 4 | 64 | 0 | 53.8 | 13.7 | 173 | 188 | 72.6 | 79.0 | 4758 | 4786 |
| 16 | 64 | 0 | 75.1 | 4.7 | 296 | 303 | 211.5 | 219.2 | 13627 | 13649 |
| 64 | 64 | 64 | 0.0 | 0.0 | 0 | 0 | 0.0 | 0.0 | 0 | 0 |
