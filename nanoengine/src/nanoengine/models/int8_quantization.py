"""int8 quantization: weights per output row, KV cache entries per (token, head) vector.

A value ``x`` is stored as ``round(x / scale)`` in int8 with ``scale = max(|x|) / 127`` for its
row or vector, and read back as ``int8 * scale``. Weights halve (fp16 → int8 + a small scale);
KV entries go from 2 bytes to 1 byte per number plus one scale per head.
"""

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional

if TYPE_CHECKING:  # qwen3 imports this module through attention_context
    from nanoengine.models.qwen3 import Qwen3ForCausalLM

INT8_MAX = 127
MIN_SCALE = 1e-8
QUANTIZED_PROJECTIONS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def quantize_vectors(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """int8 values and one scale per vector along the last dimension (scale keeps the dim)."""
    scales = values.abs().amax(dim=-1, keepdim=True).float().clamp(min=MIN_SCALE) / INT8_MAX
    quantized = torch.round(values.float() / scales).clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    return quantized, scales.to(values.dtype)


class Int8Linear(nn.Module):
    """Weight-only int8 linear layer; activations stay in the model's dtype.

    In plain PyTorch the int8 weight is turned back into the activation dtype for each matmul,
    so this saves memory, not time: fast int8 matmuls need fused kernels.
    """

    def __init__(self, weight_int8: torch.Tensor, weight_scale: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("weight_int8", weight_int8)
        self.register_buffer("weight_scale", weight_scale)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "Int8Linear":
        weight_int8, weight_scale = quantize_vectors(linear.weight.detach())
        return cls(weight_int8, weight_scale.squeeze(-1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return functional.linear(inputs, self.weight_int8.to(inputs.dtype)) * self.weight_scale


def quantize_linear_layers(model: "Qwen3ForCausalLM") -> int:
    """Replace every attention and MLP projection with ``Int8Linear``; returns how many.

    Embeddings, norms and the output head stay in full precision.
    """
    replaced = 0
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name in QUANTIZED_PROJECTIONS:
                projection = getattr(parent, name, None)
                if isinstance(projection, nn.Linear):
                    setattr(parent, name, Int8Linear.from_linear(projection))
                    replaced += 1
    return replaced
