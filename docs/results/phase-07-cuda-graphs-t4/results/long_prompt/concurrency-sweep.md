# Concurrency sweep: long_prompt on http://localhost:8000

Generated 2026-10-19 13:02 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 0 | 24.2 | 24.4 | 44 | 52 | 40.8 | 47.6 | 1317 | 1367 |
| 4 | 16 | 0 | 71.7 | 18.9 | 145 | 148 | 53.0 | 54.7 | 1782 | 1792 |
| 16 | 16 | 0 | 91.8 | 5.9 | 358 | 360 | 169.3 | 172.3 | 5570 | 5572 |
| 64 | 16 | 0 | 91.4 | 5.9 | 362 | 365 | 168.8 | 171.3 | 5595 | 5598 |
