# Harness report: http://localhost:8000

Generated 2026-10-18 08:14 UTC.

| Suite | Workload | Measured | Threshold | Outcome | Detail |
|---|---|---|---|---|---|
| prefix_reuse | shared_prefix | TTFT p50 2975 -> 1137 ms (62% lower) | >= 50% lower | PASS | 128 shared of 144 tokens |
| streaming | short_chat | 32 chunks, 32 tokens | 32 tokens, [DONE] | PASS | ok |
| latency | short_chat | TTFT p50 1214 ms, ITL p50 639.5 ms, 1.5 tok/s | no errors | PASS | 4 requests at 1 user |
