# Concurrency sweep: short_chat on http://localhost:8000

Generated 2026-10-17 19:08 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 142.6 | 145.4 | 24 | 32 | 6.9 | 7.6 | 897 | 920 |
| 4 | 64 | 0 | 478.2 | 123.1 | 39 | 45 | 8.2 | 9.3 | 1070 | 1074 |
| 16 | 64 | 0 | 1142.4 | 72.9 | 52 | 60 | 13.6 | 17.2 | 1789 | 1809 |
| 64 | 64 | 0 | 1927.9 | 31.1 | 145 | 153 | 31.7 | 40.9 | 4236 | 4243 |
