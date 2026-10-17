"""Download a Qwen3 checkpoint with ``huggingface_hub`` and load its safetensors into our model."""

import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from nanoengine.constants import (
    CONFIG_FILE_NAME,
    LM_HEAD_WEIGHT_NAME,
    MODEL_FILE_PATTERNS,
    SAFETENSORS_PATTERN,
)
from nanoengine.qwen3_model import Qwen3Config, Qwen3ForCausalLM


def download_model_files(model_id: str, cache_dir: Path | None) -> Path:
    """Config, tokenizer and weights only; later calls reuse the local cache."""
    return Path(
        snapshot_download(model_id, allow_patterns=list(MODEL_FILE_PATTERNS), cache_dir=cache_dir)
    )


def load_model_config(model_dir: Path) -> Qwen3Config:
    config_dict = json.loads((model_dir / CONFIG_FILE_NAME).read_text(encoding="utf-8"))
    return Qwen3Config.from_dict(config_dict)


def load_checkpoint_tensors(model_dir: Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for weights_file in sorted(model_dir.glob(SAFETENSORS_PATTERN)):
        tensors.update(load_file(weights_file))
    return tensors


def load_qwen3_model(model_dir: Path, device: torch.device, dtype: torch.dtype) -> Qwen3ForCausalLM:
    """Build the model without allocating memory, then move the checkpoint tensors straight in."""
    config = load_model_config(model_dir)
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)
    tensors = load_checkpoint_tensors(model_dir)
    if config.tie_word_embeddings:
        tensors.pop(LM_HEAD_WEIGHT_NAME, None)
    tensors = {name: tensor.to(device=device, dtype=dtype) for name, tensor in tensors.items()}
    model.load_state_dict(tensors, strict=True, assign=True)
    return model.eval()
