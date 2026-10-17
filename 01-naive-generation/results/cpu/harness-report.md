# Harness report: http://localhost:8000

Generated 2026-10-17 19:21 UTC.

| Suite | Workload | Measured | Threshold | Outcome | Detail |
|---|---|---|---|---|---|
| concurrency | short_chat | 1.2 -> 1.3 tok/s (1.0x) | >= 2.0x at 4 users | XFAIL | levels [1, 2, 4] |
| capacity | burst | 0/16 errors, 1.3 tok/s | error rate <= 0% | PASS | E2E p99 393533 ms |
| equivalence | equivalence | 10/10 identical (prefix match 100%) | >= 100% | PASS | all match |
| prefix_reuse | shared_prefix | TTFT p50 1556 -> 1548 ms (1% lower) | >= 50% lower | XFAIL | 128 shared of 144 tokens |
| streaming | short_chat | 32 chunks, 32 tokens | 32 tokens, [DONE] | PASS | ok |
| latency | short_chat | TTFT p50 1112 ms, ITL p50 1247.7 ms, 0.8 tok/s | no errors | PASS | 4 requests at 1 user |
