# Harness report: http://localhost:8000

Generated 2026-10-17 19:59 UTC.

| Suite | Workload | Measured | Threshold | Outcome | Detail |
|---|---|---|---|---|---|
| concurrency | short_chat | 28.3 -> 28.8 tok/s (1.0x) | >= 2.0x at 4 users | XFAIL | levels [1, 4] |
| capacity | burst | 0/64 errors, 29.1 tok/s | error rate <= 0% | PASS | E2E p99 279087 ms |
| prefix_reuse | shared_prefix | TTFT p50 57 -> 57 ms (1% lower) | >= 50% lower | XFAIL | 512 shared of 544 tokens |
| streaming | short_chat | 128 chunks, 128 tokens | 128 tokens, [DONE] | PASS | ok |
| latency | short_chat | TTFT p50 36 ms, ITL p50 34.3 ms, 28.9 tok/s | no errors | PASS | 4 requests at 1 user |
