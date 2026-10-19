# Concurrency sweep: mixed_lengths on http://localhost:8000

Generated 2026-10-18 13:57 UTC. Latencies in ms; ITL is measured between streamed chunks.

| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0 | 18.0 | 18.0 | 59 | 66 | 55.2 | 65.4 | 16299 | 27746 |
| 4 | 64 | 0 | 66.7 | 17.2 | 172 | 213 | 56.8 | 110.0 | 17263 | 29819 |
| 16 | 64 | 0 | 111.8 | 7.8 | 300 | 486 | 121.4 | 215.1 | 29209 | 73688 |
| 64 | 64 | 41 | 56.1 | 4.5 | 697 | 1189 | 220.2 | 276.4 | 27900 | 40869 |
