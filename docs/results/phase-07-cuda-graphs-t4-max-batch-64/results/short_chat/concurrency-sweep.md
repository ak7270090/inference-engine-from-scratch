# Concurrency sweep: short_chat on http://localhost:8000

Generated 2026-10-19 12:32 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 67.6 | 68.8 | 48 | 51 | 14.5 | 15.0 | 1894 | 1930 |
| 4 | 64 | 0 | 169.6 | 43.6 | 108 | 121 | 22.9 | 24.0 | 3019 | 3032 |
| 16 | 64 | 0 | 296.3 | 18.7 | 124 | 127 | 53.4 | 55.5 | 6910 | 6916 |
| 64 | 64 | 0 | 363.1 | 5.8 | 429 | 863 | 172.4 | 215.4 | 22426 | 22559 |
