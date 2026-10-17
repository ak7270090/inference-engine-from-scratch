"""Qwen3 decoder in plain PyTorch, with the same parameter names as the Hugging Face checkpoint."""

from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import nn
from torch.nn import functional


@dataclass(frozen=True)
class Qwen3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "Qwen3Config":
        return cls(**{field.name: config_dict[field.name] for field in fields(cls)})


class RMSNorm(nn.Module):
    """Scale each vector to unit root-mean-square, computed in float32 for stability."""

    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


def build_rotary_tables(
    sequence_length: int, head_dim: int, rope_theta: float, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin of every position's rotation angles, each ``[sequence_length, head_dim]``."""
    frequency_exponents = (
        torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
    )
    inverse_frequencies = 1.0 / (rope_theta**frequency_exponents)
    positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inverse_frequencies)
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def rotate_half(vectors: torch.Tensor) -> torch.Tensor:
    first_half, second_half = vectors.chunk(2, dim=-1)
    return torch.cat((-second_half, first_half), dim=-1)


def apply_rotary_embedding(
    vectors: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    return vectors * cos + rotate_half(vectors) * sin


class Qwen3Attention(nn.Module):
    """Grouped-query attention: query and key heads are RMS-normalized before RoPE."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        query_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, query_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.o_proj = nn.Linear(query_size, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        query_shape = (batch_size, sequence_length, self.num_heads, self.head_dim)
        kv_shape = (batch_size, sequence_length, self.num_kv_heads, self.head_dim)
        queries = self.q_norm(self.q_proj(hidden_states).view(query_shape)).transpose(1, 2)
        keys = self.k_norm(self.k_proj(hidden_states).view(kv_shape)).transpose(1, 2)
        values = self.v_proj(hidden_states).view(kv_shape).transpose(1, 2)
        queries = apply_rotary_embedding(queries, cos, sin)
        keys = apply_rotary_embedding(keys, cos, sin)
        heads_per_kv_head = self.num_heads // self.num_kv_heads
        keys = keys.repeat_interleave(heads_per_kv_head, dim=1)
        values = values.repeat_interleave(heads_per_kv_head, dim=1)
        attention = functional.scaled_dot_product_attention(queries, keys, values, is_causal=True)
        attention = attention.transpose(1, 2).reshape(batch_size, sequence_length, -1)
        return self.o_proj(attention)


class Qwen3MLP(nn.Module):
    """SwiGLU feed-forward: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gated = functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(gated)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), cos, sin
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(token_ids)
        cos, sin = build_rotary_tables(
            token_ids.shape[1],
            self.config.head_dim,
            self.config.rope_theta,
            hidden_states.device,
            hidden_states.dtype,
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin)
        return self.norm(hidden_states)


class Qwen3ForCausalLM(nn.Module):
    """Token IDs ``[batch, sequence]`` in, final hidden states out; ``compute_logits`` on top."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model(token_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output_weight = (
            self.model.embed_tokens.weight
            if self.config.tie_word_embeddings
            else self.lm_head.weight
        )
        return functional.linear(hidden_states, output_weight)
