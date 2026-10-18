# Concurrency sweep: long_prompt on http://localhost:8000

Generated 2026-10-17 20:11 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 0 | 9.2 | 9.2 | 107 | 133 | 108.5 | 112.2 | 3477 | 3510 |
| 4 | 16 | 0 | 9.3 | 9.3 | 10444 | 10486 | 108.0 | 110.8 | 13786 | 13840 |
