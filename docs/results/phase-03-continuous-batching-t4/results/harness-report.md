# Harness report: http://localhost:8000

Generated 2026-10-18 07:17 UTC.

| Suite | Workload | Measured | Threshold | Outcome | Detail |
|---|---|---|---|---|---|
| concurrency | short_chat | 26.8 -> 397.9 tok/s (14.9x) | >= 4.0x at 64 users | PASS | levels [1, 4, 16, 64] |
| capacity | burst | 0/64 errors, 386.9 tok/s | error rate <= 0% | PASS | E2E p99 21159 ms |
| prefix_reuse | shared_prefix | TTFT p50 49 -> 50 ms (-1% lower) | >= 50% lower | XFAIL | 512 shared of 544 tokens |
| long_prompt_interference | short_chat | ITL p99 43 -> 146 ms (3.4x) | <= 2.0x | XFAIL | 16 long prompts arriving |
| streaming | short_chat | 128 chunks, 128 tokens | 128 tokens, [DONE] | PASS | ok |
| latency | short_chat | TTFT p50 40 ms, ITL p50 37.8 ms, 26.2 tok/s | no errors | PASS | 4 requests at 1 user |
