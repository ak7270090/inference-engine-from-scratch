# Concurrency sweep: short_chat on http://localhost:8000

Generated 2026-10-18 06:06 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 20.6 | 20.6 | 51 | 58 | 48.3 | 55.3 | 6213 | 6288 |
| 4 | 64 | 0 | 20.6 | 20.6 | 18663 | 18918 | 48.3 | 55.3 | 24805 | 25107 |
