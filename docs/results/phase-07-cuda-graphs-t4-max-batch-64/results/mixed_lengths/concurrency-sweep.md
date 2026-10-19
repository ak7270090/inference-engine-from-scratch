# Concurrency sweep: mixed_lengths on http://localhost:8000

Generated 2026-10-19 12:47 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 62.8 | 64.2 | 48 | 54 | 14.7 | 22.5 | 4594 | 8403 |
| 4 | 64 | 0 | 120.1 | 30.4 | 111 | 148 | 32.8 | 66.8 | 8817 | 17823 |
| 16 | 64 | 0 | 152.3 | 10.4 | 227 | 376 | 91.0 | 167.0 | 21523 | 55864 |
| 64 | 64 | 0 | 188.6 | 5.2 | 329 | 980 | 172.2 | 317.5 | 64009 | 91586 |
