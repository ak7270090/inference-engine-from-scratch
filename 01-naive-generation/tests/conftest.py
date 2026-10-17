"""Tiny random-weight Qwen3 and a word-level tokenizer, so tests run on CPU in seconds."""

from collections.abc import Iterator

import pytest
import torch
from fastapi.testclient import TestClient
from tokenizers import Tokenizer, models, pre_tokenizers

from nanoengine.app import create_app
from nanoengine.config import EngineSettings
from nanoengine.engine_runtime import EngineRuntime, build_engine_runtime
from nanoengine.qwen3_model import Qwen3Config, Qwen3ForCausalLM
from nanoengine.text_tokenizer import TextTokenizer

TINY_VOCAB_WORDS = [f"w{index}" for index in range(63)] + ["<eos>"]
TINY_EOS_TOKEN_ID = len(TINY_VOCAB_WORDS) - 1
TINY_MODEL_NAME = "tiny-qwen3"
TINY_MAX_MODEL_LEN = 64
MODEL_SEED = 7


def build_tiny_config(tie_word_embeddings: bool = True) -> Qwen3Config:
    return Qwen3Config(
        vocab_size=len(TINY_VOCAB_WORDS),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=tie_word_embeddings,
    )


def build_tiny_model(tie_word_embeddings: bool = True) -> Qwen3ForCausalLM:
    torch.manual_seed(MODEL_SEED)
    model = Qwen3ForCausalLM(build_tiny_config(tie_word_embeddings))
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.5)
    return model.eval()


def build_tiny_tokenizer() -> TextTokenizer:
    vocabulary = {word: index for index, word in enumerate(TINY_VOCAB_WORDS)}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="w0"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return TextTokenizer(tokenizer, frozenset({TINY_EOS_TOKEN_ID}))


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def tiny_model() -> Qwen3ForCausalLM:
    return build_tiny_model()


@pytest.fixture
def tiny_tokenizer() -> TextTokenizer:
    return build_tiny_tokenizer()


@pytest.fixture
def tiny_settings() -> EngineSettings:
    return EngineSettings(
        model_id=TINY_MODEL_NAME, max_model_len=TINY_MAX_MODEL_LEN, max_waiting_requests=4
    )


@pytest.fixture
def tiny_runtime(
    tiny_model: Qwen3ForCausalLM, tiny_tokenizer: TextTokenizer, tiny_settings: EngineSettings
) -> EngineRuntime:
    return build_engine_runtime(tiny_model, tiny_tokenizer, tiny_settings)


@pytest.fixture
def api_client(tiny_settings: EngineSettings, tiny_runtime: EngineRuntime) -> Iterator[TestClient]:
    with TestClient(create_app(tiny_settings, tiny_runtime)) as client:
        yield client
