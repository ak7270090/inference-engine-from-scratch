"""Fixed names shared by the harness: endpoints, SSE markers, suite markers, file names."""

ENV_PREFIX = "HARNESS_"
ENV_FILE_VARIABLE = "HARNESS_ENV_FILE"

COMPLETIONS_PATH = "/v1/completions"
MODELS_PATH = "/v1/models"

SSE_DATA_PREFIX = "data:"
SSE_DONE_MARKER = "[DONE]"

STATUS_OK = 200

SCALE_CPU = "cpu"
SCALE_T4 = "t4"

WORKLOAD_SHORT_CHAT = "short_chat"
WORKLOAD_LONG_PROMPT = "long_prompt"
WORKLOAD_SHARED_PREFIX = "shared_prefix"
WORKLOAD_MIXED_LENGTHS = "mixed_lengths"
WORKLOAD_BURST = "burst"
WORKLOAD_EQUIVALENCE = "equivalence"

SUITE_EQUIVALENCE = "equivalence"
SUITE_STREAMING = "streaming"
SUITE_LATENCY = "latency"
SUITE_CONCURRENCY = "concurrency"
SUITE_CAPACITY = "capacity"
SUITE_PREFIX_REUSE = "prefix_reuse"
SUITE_LONG_PROMPT_INTERFERENCE = "long_prompt_interference"
ALL_SUITES = frozenset(
    {
        SUITE_EQUIVALENCE,
        SUITE_STREAMING,
        SUITE_LATENCY,
        SUITE_CONCURRENCY,
        SUITE_CAPACITY,
        SUITE_PREFIX_REUSE,
        SUITE_LONG_PROMPT_INTERFERENCE,
    }
)

OUTCOME_PASS = "PASS"
OUTCOME_FAIL = "FAIL"
OUTCOME_XFAIL = "XFAIL"
OUTCOME_XPASS = "XPASS"
OUTCOME_SKIPPED = "SKIP"

REPORT_FILE_NAME = "harness-report.md"
SWEEP_REPORT_FILE_NAME = "concurrency-sweep.md"
SWEEP_CHART_FILE_NAME = "concurrency-sweep.png"
METRICS_JSON_FILE_NAME = "metrics.json"

CORPUS_TEXT = (
    "the river carried stories from the mountains to the sea while traders in the old market "
    "counted spices and silver under lanterns that swayed in the evening wind and children "
    "chased kites across rooftops as the monsoon clouds gathered over the quiet harbour"
)
