# Harness report: http://localhost:8000

Generated 2026-10-18 08:11 UTC.

| Suite | Workload | Measured | Threshold | Outcome | Detail |
|---|---|---|---|---|---|
| concurrency | short_chat | 27.3 -> 417.6 tok/s (15.3x) | >= 4.0x at 64 users | PASS | levels [1, 4, 16, 64] |
| capacity | burst | 0/64 errors, 416.1 tok/s | error rate <= 0% | PASS | E2E p99 19678 ms |
| prefix_reuse | shared_prefix | TTFT p50 100 -> 100 ms (0% lower) | >= 50% lower | XFAIL | 512 shared of 544 tokens |
| long_prompt_interference | short_chat | ITL p99 42 -> 39 ms (0.9x) | <= 2.0x | XPASS | 16 long prompts arriving |
| streaming | short_chat | 128 chunks, 128 tokens | 128 tokens, [DONE] | PASS | ok |
| latency | short_chat | TTFT p50 88 ms, ITL p50 35.7 ms, 27.8 tok/s | no errors | PASS | 4 requests at 1 user |
