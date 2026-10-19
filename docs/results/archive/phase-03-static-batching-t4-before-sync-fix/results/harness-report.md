# Harness report: http://localhost:8000

Generated 2026-10-18 06:59 UTC.

| Suite | Workload | Measured | Threshold | Outcome | Detail |
|---|---|---|---|---|---|
| concurrency | short_chat | 22.3 -> 348.7 tok/s (15.6x) | >= 4.0x at 64 users | PASS | levels [1, 4, 16, 64] |
| capacity | burst | 0/64 errors, 350.0 tok/s | error rate <= 0% | PASS | E2E p99 23396 ms |
| prefix_reuse | shared_prefix | TTFT p50 110 -> 110 ms (0% lower) | >= 50% lower | XFAIL | 512 shared of 544 tokens |
| long_prompt_interference | short_chat | ITL p99 52 -> 50 ms (1.0x) | <= 2.0x | XPASS | 16 long prompts arriving |
| streaming | short_chat | 128 chunks, 128 tokens | 128 tokens, [DONE] | PASS | ok |
| latency | short_chat | TTFT p50 95 ms, ITL p50 42.5 ms, 23.3 tok/s | no errors | PASS | 4 requests at 1 user |
