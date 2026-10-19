# Concurrency sweep: short_chat on http://localhost:8000

Generated 2026-10-18 06:36 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 22.2 | 22.2 | 48 | 57 | 44.8 | 52.5 | 5764 | 5936 |
| 4 | 64 | 0 | 88.4 | 21.9 | 52 | 101 | 44.3 | 56.5 | 5866 | 6036 |
| 16 | 64 | 0 | 337.3 | 21.0 | 143 | 157 | 46.2 | 56.9 | 6156 | 6236 |
| 64 | 64 | 0 | 332.1 | 20.9 | 6384 | 18753 | 47.4 | 54.0 | 12438 | 24663 |
