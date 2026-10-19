# Concurrency sweep: long_prompt on http://localhost:8000

Generated 2026-10-19 14:33 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 0 | 15.8 | 15.8 | 67 | 73 | 63.0 | 67.5 | 2026 | 2068 |
| 4 | 16 | 0 | 45.0 | 11.8 | 220 | 224 | 84.8 | 91.2 | 2838 | 2851 |
| 16 | 16 | 0 | 67.3 | 4.4 | 475 | 477 | 231.3 | 235.2 | 7602 | 7604 |
| 64 | 16 | 0 | 67.0 | 4.3 | 466 | 468 | 231.1 | 235.5 | 7637 | 7639 |
