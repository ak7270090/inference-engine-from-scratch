# Harness report: http://localhost:8000

Generated 2026-10-18 06:28 UTC.

| Suite | Workload | Measured | Threshold | Outcome | Detail |
|---|---|---|---|---|---|
| concurrency | short_chat | 21.7 -> 332.3 tok/s (15.3x) | >= 4.0x at 64 users | PASS | levels [1, 4, 16, 64] |
| capacity | burst | 0/64 errors, 330.2 tok/s | error rate <= 0% | PASS | E2E p99 24800 ms |
| prefix_reuse | shared_prefix | TTFT p50 61 -> 61 ms (-0% lower) | >= 50% lower | XFAIL | 512 shared of 544 tokens |
| long_prompt_interference | short_chat | ITL p99 51 -> 169 ms (3.3x) | <= 2.0x | XFAIL | 16 long prompts arriving |
| streaming | short_chat | 128 chunks, 128 tokens | 128 tokens, [DONE] | PASS | ok |
| latency | short_chat | TTFT p50 47 ms, ITL p50 44.6 ms, 22.3 tok/s | no errors | PASS | 4 requests at 1 user |
