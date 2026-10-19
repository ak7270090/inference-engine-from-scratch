# Concurrency sweep: short_chat on http://localhost:8000

Generated 2026-10-18 18:36 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 17.3 | 17.3 | 60 | 65 | 57.4 | 67.6 | 7382 | 7679 |
| 4 | 64 | 0 | 66.5 | 16.9 | 178 | 187 | 58.6 | 71.9 | 7677 | 7918 |
| 16 | 64 | 0 | 262.7 | 16.7 | 184 | 189 | 59.5 | 68.0 | 7782 | 7817 |
| 64 | 64 | 64 | 0.0 | 0.0 | 0 | 0 | 0.0 | 0.0 | 0 | 0 |
