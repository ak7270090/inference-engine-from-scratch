# Concurrency sweep: short_chat on http://localhost:8000

Generated 2026-10-18 11:53 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 23.6 | 23.6 | 46 | 51 | 42.1 | 48.7 | 5429 | 5575 |
| 4 | 64 | 0 | 90.6 | 22.9 | 131 | 135 | 43.2 | 50.5 | 5660 | 5748 |
| 16 | 64 | 0 | 349.3 | 22.2 | 144 | 149 | 44.7 | 53.0 | 5865 | 5897 |
| 64 | 64 | 0 | 602.1 | 9.7 | 357 | 564 | 102.9 | 143.0 | 13489 | 13602 |
