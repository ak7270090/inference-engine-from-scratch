# Concurrency sweep: short_chat on http://localhost:8000

Generated 2026-10-17 19:01 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 8 | 0 | 1.2 | 1.3 | 676 | 806 | 771.6 | 1908.4 | 24874 | 35207 |
| 2 | 8 | 0 | 1.2 | 1.1 | 28551 | 31034 | 830.2 | 1374.3 | 55869 | 60191 |
| 4 | 8 | 0 | 1.3 | 1.3 | 74681 | 76435 | 774.9 | 1144.4 | 99170 | 102907 |
