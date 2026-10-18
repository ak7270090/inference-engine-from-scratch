# Harness report: http://localhost:8000

Generated 2026-10-18 05:40 UTC.

| Suite | Workload | Measured | Threshold | Outcome | Detail |
|---|---|---|---|---|---|
| concurrency | short_chat | 23.6 -> 993.8 tok/s (42.1x) | >= 4.0x at 64 users | PASS | levels [1, 4, 16, 64] |
| capacity | burst | 0/64 errors, 992.3 tok/s | error rate <= 0% | PASS | E2E p99 8254 ms |
| prefix_reuse | shared_prefix | TTFT p50 57 -> 57 ms (0% lower) | >= 50% lower | XFAIL | 512 shared of 544 tokens |
| long_prompt_interference | short_chat | ITL p99 50 -> 156 ms (3.2x) | <= 2.0x | XFAIL | 16 long prompts arriving |
| streaming | short_chat | 128 chunks, 128 tokens | 128 tokens, [DONE] | PASS | ok |
| latency | short_chat | TTFT p50 44 ms, ITL p50 41.6 ms, 23.9 tok/s | no errors | PASS | 4 requests at 1 user |
