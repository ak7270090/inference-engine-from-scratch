# Concurrency sweep: short_chat on http://localhost:8000

Generated 2026-10-18 18:26 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 24.7 | 24.7 | 45 | 49 | 40.1 | 46.5 | 5184 | 5266 |
| 4 | 64 | 0 | 94.5 | 24.0 | 126 | 137 | 41.3 | 49.2 | 5413 | 5521 |
| 16 | 64 | 0 | 366.4 | 23.2 | 134 | 146 | 42.5 | 53.2 | 5593 | 5633 |
| 64 | 64 | 64 | 0.0 | 0.0 | 0 | 0 | 0.0 | 0.0 | 0 | 0 |
