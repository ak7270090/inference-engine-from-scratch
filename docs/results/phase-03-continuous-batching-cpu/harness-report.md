# Harness report: http://localhost:8000

Generated 2026-10-18 05:28 UTC.

| Suite | Workload | Measured | Threshold | Outcome | Detail |
|---|---|---|---|---|---|
| equivalence | equivalence | 10/10 identical (prefix match 100%) | >= 100% | PASS | all match |
| streaming | short_chat | 32 chunks, 32 tokens | 32 tokens, [DONE] | PASS | ok |
| latency | short_chat | TTFT p50 692 ms, ITL p50 202.2 ms, 4.9 tok/s | no errors | PASS | 4 requests at 1 user |
