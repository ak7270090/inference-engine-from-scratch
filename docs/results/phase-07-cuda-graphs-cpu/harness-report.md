# Harness report: http://localhost:8000

Generated 2026-10-18 08:09 UTC.

| Suite | Workload | Measured | Threshold | Outcome | Detail |
|---|---|---|---|---|---|
| equivalence | equivalence | 10/10 identical (prefix match 100%) | >= 100% | PASS | all match |
| prefix_reuse | shared_prefix | TTFT p50 2247 -> 701 ms (69% lower) | >= 50% lower | PASS | 128 shared of 144 tokens |
| streaming | short_chat | 32 chunks, 32 tokens | 32 tokens, [DONE] | PASS | ok |
| latency | short_chat | TTFT p50 938 ms, ITL p50 297.8 ms, 3.2 tok/s | no errors | PASS | 4 requests at 1 user |
