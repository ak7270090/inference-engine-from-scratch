import os

import pytest
import torch
from transformers import Qwen3Config as HuggingFaceQwen3Config
from transformers import Qwen3ForCausalLM as HuggingFaceQwen3ForCausalLM

from nanoengine.constants import DEFAULT_MODEL_ID
from nanoengine.qwen3_model import build_rotary_tables, rotate_half
from nanoengine.weight_loader import download_model_files, load_qwen3_model
from tests.conftest import build_tiny_config, build_tiny_model

LOGITS_TOLERANCE = 1e-4
REAL_WEIGHTS_VARIABLE = "NANOENGINE_TEST_REAL_WEIGHTS"


def build_hugging_face_twin(tie_word_embeddings: bool) -> tuple:
    our_model = build_tiny_model(tie_word_embeddings)
    config = build_tiny_config(tie_word_embeddings)
    their_config = HuggingFaceQwen3Config(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        tie_word_embeddings=tie_word_embeddings,
        attn_implementation="eager",
    )
    their_model = HuggingFaceQwen3ForCausalLM(their_config).eval()
    their_model.load_state_dict(our_model.state_dict(), strict=False)
    if tie_word_embeddings:
        their_model.tie_weights()
    return our_model, their_model


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_logits_match_transformers_for_random_weights(tie_word_embeddings: bool) -> None:
    our_model, their_model = build_hugging_face_twin(tie_word_embeddings)
    token_ids = torch.randint(0, our_model.config.vocab_size, (2, 11))

    with torch.inference_mode():
        our_logits = our_model.compute_logits(our_model(token_ids))
        their_logits = their_model(token_ids).logits

    torch.testing.assert_close(our_logits, their_logits, atol=LOGITS_TOLERANCE, rtol=0)


def test_later_tokens_do_not_change_earlier_logits() -> None:
    model = build_tiny_model()
    token_ids = torch.randint(0, model.config.vocab_size, (1, 9))

    with torch.inference_mode():
        full_logits = model.compute_logits(model(token_ids))
        prefix_logits = model.compute_logits(model(token_ids[:, :5]))

    torch.testing.assert_close(full_logits[:, :5], prefix_logits, atol=1e-5, rtol=0)


def test_rotary_tables_start_with_no_rotation_at_position_zero() -> None:
    cos, sin = build_rotary_tables(4, 8, 10000.0, torch.device("cpu"), torch.float32)

    assert cos.shape == sin.shape == (4, 8)
    torch.testing.assert_close(cos[0], torch.ones(8))
    torch.testing.assert_close(sin[0], torch.zeros(8))


def test_rotate_half_maps_first_half_to_second() -> None:
    vectors = torch.tensor([1.0, 2.0, 3.0, 4.0])

    assert rotate_half(vectors).tolist() == [-3.0, -4.0, 1.0, 2.0]


@pytest.mark.skipif(
    not os.environ.get(REAL_WEIGHTS_VARIABLE), reason=f"set {REAL_WEIGHTS_VARIABLE}=1 to download"
)
def test_logits_match_transformers_for_real_qwen3_weights() -> None:
    model_dir = download_model_files(DEFAULT_MODEL_ID, None)
    our_model = load_qwen3_model(model_dir, torch.device("cpu"), torch.float32)
    their_model = HuggingFaceQwen3ForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    token_ids = torch.tensor([[785, 6722, 315, 9625, 374]])

    with torch.inference_mode():
        our_logits = our_model.compute_logits(our_model(token_ids))
        their_logits = their_model(token_ids).logits

    torch.testing.assert_close(our_logits, their_logits, atol=1e-3, rtol=0)
    assert torch.equal(our_logits.argmax(-1), their_logits.argmax(-1))
