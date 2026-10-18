# Concurrency sweep: short_chat on http://localhost:8000

Generated 2026-10-17 20:09 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 28.3 | 28.3 | 37 | 41 | 35.2 | 42.7 | 4522 | 4751 |
| 4 | 64 | 0 | 27.8 | 27.8 | 13805 | 14095 | 35.7 | 42.2 | 18350 | 18709 |
