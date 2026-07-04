from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def exists(value) -> bool:
    return value is not None


def default(value, default_value):
    return value if exists(value) else default_value


def to_pairwise_mask(mask: Tensor) -> Tensor:
    return mask[:, :, None] & mask[:, None, :]


class SwiGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x, gates = x.chunk(2, dim=-1)
        return F.silu(gates) * x


class Transition(nn.Module):
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


class StructuredDropout(nn.Module):
    def __init__(self, prob: float, dropout_type: str | None = None):
        super().__init__()
        self.dropout = nn.Dropout(prob)
        self.dropout_type = dropout_type

    def forward(self, x: Tensor) -> Tensor:
        if self.dropout_type is None:
            return self.dropout(x)

        if x.ndim != 4:
            raise ValueError("structured dropout expects a 4D tensor [b, n, n, d]")

        if self.dropout_type == "row":
            b, _, n, d = x.shape
            keep = x.new_ones((b, 1, n, d))
        elif self.dropout_type == "col":
            b, n, _, d = x.shape
            keep = x.new_ones((b, n, 1, d))
        else:
            raise ValueError(f"unknown dropout_type: {self.dropout_type}")

        keep = self.dropout(keep)
        return x * keep


class PreLayerNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
    """Minimal multi-head attention used by triangle attention."""

    def __init__(
        self,
        dim: int,
        *,
        dim_head: int = 32,
        heads: int = 4,
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


class TriangleMultiplication(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        dim_hidden: int | None = None,
        mix: str = "incoming",
        dropout: float = 0.0,
        dropout_type: str | None = None,
    ):
        super().__init__()
        dim_hidden = default(dim_hidden, dim)

        self.left_right_proj = nn.Sequential(
            nn.Linear(dim, dim_hidden * 4, bias=False),
            nn.GLU(dim=-1),
        )
        self.out_gate = nn.Linear(dim, dim_hidden, bias=False)
        self.to_out_norm = nn.LayerNorm(dim_hidden)
        self.to_out = nn.Sequential(
            nn.Linear(dim_hidden, dim, bias=False),
            StructuredDropout(dropout, dropout_type=dropout_type),
        )

        if mix not in {"incoming", "outgoing"}:
            raise ValueError("mix must be 'incoming' or 'outgoing'")
        self.mix = mix

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        if exists(mask):
            pair_mask = to_pairwise_mask(mask).unsqueeze(-1)
        else:
            pair_mask = None

        left, right = self.left_right_proj(x).chunk(2, dim=-1)

        if exists(pair_mask):
            left = left * pair_mask
            right = right * pair_mask

        if self.mix == "outgoing":
            out = torch.einsum("b i k d, b j k d -> b i j d", left, right)
        else:
            out = torch.einsum("b k j d, b k i d -> b i j d", left, right)

        out = self.to_out_norm(out)
        out_gate = torch.sigmoid(self.out_gate(x))
        return self.to_out(out) * out_gate


class TriangleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        heads: int,
        dim_head: int = 32,
        node_type: str = "starting",
        dropout: float = 0.0,
        dropout_type: str | None = None,
    ):
        super().__init__()

        if node_type not in {"starting", "ending"}:
            raise ValueError("node_type must be 'starting' or 'ending'")

        self.need_transpose = node_type == "ending"
        self.attn = Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.dropout = StructuredDropout(dropout, dropout_type=dropout_type)
        self.to_attn_bias = nn.Linear(dim, heads, bias=False)

    def forward(self, pairwise_repr: Tensor, mask: Tensor | None = None) -> Tensor:
        x = pairwise_repr.transpose(1, 2) if self.need_transpose else pairwise_repr
        b, n, _, d = x.shape

        attn_bias = self.to_attn_bias(x).permute(0, 3, 1, 2)
        attn_bias = attn_bias.unsqueeze(1).repeat(1, n, 1, 1, 1)
        attn_bias = attn_bias.reshape(b * n, attn_bias.shape[2], attn_bias.shape[3], attn_bias.shape[4])

        if exists(mask):
            attn_mask = mask.unsqueeze(1).repeat(1, n, 1).reshape(b * n, n)
        else:
            attn_mask = None

        x = x.reshape(b * n, n, d)
        out = self.attn(x, mask=attn_mask, attn_bias=attn_bias)
        out = out.view(b, n, n, d)

        if self.need_transpose:
            out = out.transpose(1, 2)

        return self.dropout(out)


class TriangleUpdate(nn.Module):
    """Standalone AF3-style triangle update block.

    Input:
        pairwise_repr: [b, n, n, d]

    Output:
        pairwise_repr: [b, n, n, d]
    """

    def __init__(
        self,
        *,
        dim_pairwise: int,
        tri_mult_dim_hidden: int | None = None,
        tri_attn_dim_head: int = 32,
        tri_attn_heads: int = 4,
        dropout_row_prob: float = 0.25,
        dropout_col_prob: float = 0.25,
        transition_expansion_factor: float = 2.0,
    ):
        super().__init__()

        pre_ln = partial(PreLayerNorm, dim_pairwise)

        self.tri_mult_outgoing = pre_ln(
            TriangleMultiplication(
                dim_pairwise,
                dim_hidden=tri_mult_dim_hidden,
                mix="outgoing",
                dropout=dropout_row_prob,
                dropout_type="row",
            )
        )
        self.tri_mult_incoming = pre_ln(
            TriangleMultiplication(
                dim_pairwise,
                dim_hidden=tri_mult_dim_hidden,
                mix="incoming",
                dropout=dropout_row_prob,
                dropout_type="row",
            )
        )
        self.tri_attn_starting = pre_ln(
            TriangleAttention(
                dim_pairwise,
                heads=tri_attn_heads,
                dim_head=tri_attn_dim_head,
                node_type="starting",
                dropout=dropout_row_prob,
                dropout_type="row",
            )
        )
        self.tri_attn_ending = pre_ln(
            TriangleAttention(
                dim_pairwise,
                heads=tri_attn_heads,
                dim_head=tri_attn_dim_head,
                node_type="ending",
                dropout=dropout_col_prob,
                dropout_type="col",
            )
        )
        self.pairwise_transition = pre_ln(
            Transition(dim_pairwise, expansion_factor=transition_expansion_factor)
        )

    def forward(self, pairwise_repr: Tensor, mask: Tensor | None = None) -> Tensor:
        pairwise_repr = pairwise_repr + self.tri_mult_outgoing(pairwise_repr, mask=mask)
        pairwise_repr = pairwise_repr + self.tri_mult_incoming(pairwise_repr, mask=mask)
        pairwise_repr = pairwise_repr + self.tri_attn_starting(pairwise_repr, mask=mask)
        pairwise_repr = pairwise_repr + self.tri_attn_ending(pairwise_repr, mask=mask)
        pairwise_repr = pairwise_repr + self.pairwise_transition(pairwise_repr)
        return pairwise_repr


if __name__ == "__main__":
    model = TriangleUpdate(
        dim_pairwise=128,
        tri_mult_dim_hidden=128,
        tri_attn_dim_head=32,
        tri_attn_heads=4,
    )

    pairwise_repr = torch.randn(2, 16, 16, 128)
    mask = torch.ones(2, 16, dtype=torch.bool)

    out = model(pairwise_repr, mask=mask)
    print("input shape:", tuple(pairwise_repr.shape))
    print("output shape:", tuple(out.shape))
