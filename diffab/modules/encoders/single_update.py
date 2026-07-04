from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def exists(value) -> bool:
    return value is not None


class SwiGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x, gates = x.chunk(2, dim=-1)
        return F.silu(gates) * x


class Transition(nn.Module):
    """Token-wise feedforward block used after pair-biased attention."""

    def __init__(self, dim: int, expansion_factor: float = 2.0):
        super().__init__()
        dim_inner = int(dim * expansion_factor)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim_inner * 2, bias=False),
            SwiGLU(),
            nn.Linear(dim_inner, dim, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ff(x)


class PreLayerNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
    """Minimal multi-head self-attention."""

    def __init__(
        self,
        dim: int,
        *,
        dim_head: int = 64,
        heads: int = 8,
        dropout: float = 0.0,
        gate_output: bool = True,
        query_bias: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        dim_inner = heads * dim_head

        self.to_q = nn.Linear(dim, dim_inner, bias=query_bias)
        self.to_kv = nn.Linear(dim, dim_inner * 2, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.to_gates = (
            nn.Sequential(nn.Linear(dim, dim_inner, bias=False), nn.Sigmoid())
            if gate_output
            else None
        )

    def forward(
        self,
        seq: Tensor,
        *,
        mask: Tensor | None = None,
        attn_bias: Tensor | None = None,
    ) -> Tensor:
        b, n, _ = seq.shape

        q = self.to_q(seq)
        k, v = self.to_kv(seq).chunk(2, dim=-1)

        q = q.view(b, n, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(b, n, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(b, n, self.heads, self.dim_head).transpose(1, 2)

        sim = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if exists(attn_bias):
            sim = sim + attn_bias

        if exists(mask):
            pair_mask = mask[:, None, :, None] & mask[:, None, None, :]
            sim = sim.masked_fill(~pair_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, self.heads * self.dim_head)

        if exists(self.to_gates):
            out = out * self.to_gates(seq)

        return self.to_out(out)


class AttentionPairBias(nn.Module):
    """AF3-style single update: self-attention with pairwise-derived attention bias."""

    def __init__(
        self,
        *,
        dim_single: int,
        dim_pairwise: int,
        heads: int = 16,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn = Attention(
            dim=dim_single,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )
        self.to_attn_bias_norm = nn.LayerNorm(dim_pairwise)
        self.to_attn_bias = nn.Linear(dim_pairwise, heads, bias=False)

    def forward(
        self,
        single_repr: Tensor,
        *,
        pairwise_repr: Tensor,
        mask: Tensor | None = None,
        attn_bias: Tensor | None = None,
    ) -> Tensor:
        pairwise_bias = self.to_attn_bias(self.to_attn_bias_norm(pairwise_repr))
        pairwise_bias = pairwise_bias.permute(0, 3, 1, 2)

        if exists(attn_bias):
            pairwise_bias = pairwise_bias + attn_bias[:, None, :, :]

        return self.attn(single_repr, mask=mask, attn_bias=pairwise_bias)


class SingleUpdate(nn.Module):
    """Standalone AF3-style single representation update block.

    Inputs:
        single_repr:   [b, n, ds]
        pairwise_repr: [b, n, n, dp]

    Output:
        single_repr:   [b, n, ds]
    """

    def __init__(
        self,
        *,
        dim_single: int,
        dim_pairwise: int,
        pair_bias_attn_heads: int = 16,
        pair_bias_attn_dim_head: int = 64,
        dropout: float = 0.25,
        transition_expansion_factor: float = 2.0,
    ):
        super().__init__()
        self.pair_bias_attn = PreLayerNorm(
            dim_single,
            AttentionPairBias(
                dim_single=dim_single,
                dim_pairwise=dim_pairwise,
                heads=pair_bias_attn_heads,
                dim_head=pair_bias_attn_dim_head,
                dropout=dropout,
            ),
        )
        self.single_transition = PreLayerNorm(
            dim_single,
            Transition(dim_single, expansion_factor=transition_expansion_factor),
        )

    def forward(
        self,
        single_repr: Tensor,
        pairwise_repr: Tensor,
        mask: Tensor | None = None,
        attn_bias: Tensor | None = None,
    ) -> Tensor:
        single_repr = single_repr + self.pair_bias_attn(
            single_repr,
            pairwise_repr=pairwise_repr,
            mask=mask,
            attn_bias=attn_bias,
        )
        single_repr = single_repr + self.single_transition(single_repr)
        return single_repr


if __name__ == "__main__":
    model = SingleUpdate(
        dim_single=384,
        dim_pairwise=128,
        pair_bias_attn_heads=8,
        pair_bias_attn_dim_head=32,
    )

    single_repr = torch.randn(2, 16, 384)
    pairwise_repr = torch.randn(2, 16, 16, 128)
    mask = torch.ones(2, 16, dtype=torch.bool)

    out = model(single_repr, pairwise_repr, mask=mask)

    print("single input shape:", tuple(single_repr.shape))
    print("pairwise input shape:", tuple(pairwise_repr.shape))
    print("single output shape:", tuple(out.shape))
