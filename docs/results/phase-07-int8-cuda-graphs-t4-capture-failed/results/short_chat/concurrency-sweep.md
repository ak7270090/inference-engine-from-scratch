# Concurrency sweep: short_chat on http://localhost:8000

Generated 2026-10-18 13:27 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 17.8 | 17.8 | 59 | 66 | 55.6 | 64.9 | 7175 | 7317 |
| 4 | 64 | 0 | 68.7 | 17.5 | 172 | 180 | 56.9 | 70.2 | 7443 | 7578 |
| 16 | 64 | 0 | 212.2 | 13.4 | 181 | 187 | 74.4 | 81.9 | 9644 | 9668 |
| 64 | 64 | 0 | 283.1 | 4.5 | 582 | 1137 | 220.8 | 274.9 | 28739 | 28927 |
