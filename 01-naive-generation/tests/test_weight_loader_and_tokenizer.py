import json
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file

from nanoengine import weight_loader
from nanoengine.constants import (
    CONFIG_FILE_NAME,
    GENERATION_CONFIG_FILE_NAME,
    LM_HEAD_WEIGHT_NAME,
    TOKENIZER_FILE_NAME,
)
from nanoengine.text_tokenizer import IncrementalDecoder, TextTokenizer, load_text_tokenizer
from nanoengine.weight_loader import download_model_files, load_qwen3_model
from tests.conftest import TINY_EOS_TOKEN_ID, build_tiny_model, build_tiny_tokenizer


def write_tiny_checkpoint(model_dir: Path) -> torch.nn.Module:
    model = build_tiny_model()
    config = {**asdict(model.config), "eos_token_id": TINY_EOS_TOKEN_ID}
    (model_dir / CONFIG_FILE_NAME).write_text(json.dumps(config), encoding="utf-8")
    (model_dir / GENERATION_CONFIG_FILE_NAME).write_text(
        json.dumps({"eos_token_id": [TINY_EOS_TOKEN_ID, 5]}), encoding="utf-8"
    )
    tensors = {name: tensor.contiguous() for name, tensor in model.state_dict().items()}
    tensors[LM_HEAD_WEIGHT_NAME] = model.model.embed_tokens.weight.detach().clone()
    save_file(tensors, model_dir / "model.safetensors")
    return model


def test_loaded_model_gives_the_same_logits_as_the_saved_one(tmp_path: Path) -> None:
    saved_model = write_tiny_checkpoint(tmp_path)
    token_ids = torch.tensor([[1, 2, 3, 4]])

    loaded_model = load_qwen3_model(tmp_path, torch.device("cpu"), torch.float32)

    with torch.inference_mode():
        torch.testing.assert_close(
            loaded_model.compute_logits(loaded_model(token_ids)),
            saved_model.compute_logits(saved_model(token_ids)),
        )


class RecordingDownloader:
    def __init__(self, local_dir: Path) -> None:
        self.local_dir = local_dir
        self.calls: list[dict] = []

    def __call__(self, model_id: str, allow_patterns: list[str], cache_dir: Path | None) -> str:
        self.calls.append({"model_id": model_id, "allow_patterns": allow_patterns})
        return str(self.local_dir)


class ScriptedTokenizer(TextTokenizer):
    """Decodes from a fixed table, to imitate byte-level tokens that split a character."""

    def __init__(self, decoded_texts: dict[tuple[int, ...], str]) -> None:
        self._decoded_texts = decoded_texts

    def decode(self, token_ids: list[int]) -> str:
        return self._decoded_texts[tuple(token_ids)]


def test_download_asks_only_for_config_tokenizer_and_weights(monkeypatch, tmp_path: Path) -> None:
    downloader = RecordingDownloader(tmp_path)
    monkeypatch.setattr(weight_loader, "snapshot_download", downloader)

    model_dir = download_model_files("org/model", None)

    assert model_dir == tmp_path
    assert downloader.calls[0]["model_id"] == "org/model"
    assert "*.safetensors" in downloader.calls[0]["allow_patterns"]


def test_stop_tokens_come_from_both_config_files(tmp_path: Path) -> None:
    write_tiny_checkpoint(tmp_path)
    build_tiny_tokenizer()._tokenizer.save(str(tmp_path / TOKENIZER_FILE_NAME))

    tokenizer = load_text_tokenizer(tmp_path)

    assert tokenizer.stop_token_ids == frozenset({TINY_EOS_TOKEN_ID, 5})
    assert tokenizer.encode("w3 w4") == [3, 4]


def test_incremental_decoder_holds_back_a_partial_character() -> None:
    tokenizer = ScriptedTokenizer({(1,): "a", (1, 2): "a�", (1, 2, 3): "aé"})
    decoder = IncrementalDecoder(tokenizer)

    assert [decoder.add_token(token_id) for token_id in (1, 2, 3)] == ["a", "", "é"]
    assert decoder.flush() == ""
