"""Fixed names: endpoints, SSE markers, OpenAI object names, checkpoint file names."""

import torch

PHASE_NAME = "01-naive-generation"
ENV_PREFIX = "NANOENGINE_"
ENV_FILE_VARIABLE = "NANOENGINE_ENV_FILE"
DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"

HEALTH_PATH = "/healthz"
MODELS_PATH = "/v1/models"
COMPLETIONS_PATH = "/v1/completions"
METRICS_PATH = "/metrics"

SSE_MEDIA_TYPE = "text/event-stream"
SSE_DATA_PREFIX = "data: "
SSE_EVENT_END = "\n\n"
SSE_DONE_MESSAGE = "data: [DONE]\n\n"

COMPLETION_ID_PREFIX = "cmpl-"
TEXT_COMPLETION_OBJECT = "text_completion"
MODEL_OBJECT = "model"
LIST_OBJECT = "list"
MODEL_OWNER = "nanoengine"

FINISH_REASON_STOP = "stop"
FINISH_REASON_LENGTH = "length"

CONFIG_FILE_NAME = "config.json"
GENERATION_CONFIG_FILE_NAME = "generation_config.json"
TOKENIZER_FILE_NAME = "tokenizer.json"
SAFETENSORS_PATTERN = "*.safetensors"
MODEL_FILE_PATTERNS = (
    CONFIG_FILE_NAME,
    GENERATION_CONFIG_FILE_NAME,
    TOKENIZER_FILE_NAME,
    SAFETENSORS_PATTERN,
)
LM_HEAD_WEIGHT_NAME = "lm_head.weight"
EOS_TOKEN_ID_KEY = "eos_token_id"

UTF8_REPLACEMENT_CHARACTER = "�"

TORCH_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

STEP_LATENCY_BUCKETS_SECONDS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
