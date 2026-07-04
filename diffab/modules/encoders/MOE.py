import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from transformers.activations import ACT2FN


class MiniMindABConfig:
    def __init__(
        self,
        input_dim: int = 128,
        vocab_size: int = 128,
        hidden_size: int = 512,
        intermediate_size: Optional[int] = None,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        max_position_embeddings: int = 4096,
        hidden_act: str = "silu",
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 1_000_000.0,
        flash_attn: bool = True,
        decoder_causal: bool = True,
        num_fragment_types: int = 3,
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
    ):
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.dropout = dropout
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.flash_attn = flash_attn
        self.decoder_causal = decoder_causal
        self.num_fragment_types = num_fragment_types
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int, rope_base: float):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, seq_len, num_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(batch, seq_len, num_heads, n_rep, head_dim).reshape(
        batch, seq_len, num_heads * n_rep, head_dim
    )


class SelfAttention(nn.Module):
    def __init__(self, config: MiniMindABConfig):
        super().__init__()
        self.num_key_value_heads = (
            config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        )
        assert config.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(F, "scaled_dot_product_attention") and config.flash_attn

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        xq = self.q_proj(x).view(batch_size, seq_len, self.n_local_heads, self.head_dim)
        xk = self.k_proj(x).view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        attn_mask = _prepare_attention_mask(attention_mask, batch_size, seq_len, seq_len, xq.device)
        if self.flash and attn_mask is None:
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if is_causal:
                causal = torch.triu(
                    torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                    diagonal=1,
                )
                scores = scores + causal
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, -1e9)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.resid_dropout(self.o_proj(output))


class CrossAttention(nn.Module):
    def __init__(self, config: MiniMindABConfig):
        super().__init__()
        self.num_key_value_heads = (
            config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        )
        assert config.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, tgt_len, _ = x.shape
        src_len = memory.shape[1]
        xq = self.q_proj(x).view(batch_size, tgt_len, self.n_local_heads, self.head_dim).transpose(1, 2)
        xk = self.k_proj(memory).view(batch_size, src_len, self.n_local_kv_heads, self.head_dim)
        xv = self.v_proj(memory).view(batch_size, src_len, self.n_local_kv_heads, self.head_dim)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_mask = _prepare_attention_mask(attention_mask, batch_size, tgt_len, src_len, xq.device)
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, -1e9)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = self.attn_dropout(scores)
        output = scores @ xv
        output = output.transpose(1, 2).reshape(batch_size, tgt_len, -1)
        return self.resid_dropout(self.o_proj(output))


class FeedForward(nn.Module):
    def __init__(self, config: MiniMindABConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 63) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    def __init__(self, config: MiniMindABConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_dim)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func != "softmax":
            raise NotImplementedError(f"Unsupported MoE scoring function: {self.scoring_func}")
        scores = logits.softmax(dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        if self.top_k > 1 and self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        if self.training and self.alpha > 0.0:
            topk_idx_for_aux_loss = topk_idx.view(batch_size, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores.view(batch_size, seq_len, -1)
                ce = torch.zeros(batch_size, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(batch_size, seq_len * self.top_k, device=hidden_states.device),
                ).div_(seq_len * self.top_k / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.reshape(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                pi = scores.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(())
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindABConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(config.n_routed_experts)])
        self.gate = MoEGate(config)
        self.aux_loss = torch.tensor(0.0)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([FeedForward(config) for _ in range(config.n_shared_experts)])
        else:
            self.shared_experts = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        orig_shape = x.shape
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.reshape(-1, x.shape[-1])
        flat_topk_idx = topk_idx.reshape(-1)
        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x)
            for expert_id, expert in enumerate(self.experts):
                expert_mask = flat_topk_idx == expert_id
                expert_out = expert(x[expert_mask])
                if expert_out.shape[0] > 0:
                    y[expert_mask] = expert_out.to(y.dtype)
                else:
                    y[expert_mask] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1).view(*orig_shape)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.reshape(-1, 1)).view(*orig_shape)

        if self.shared_experts is not None:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x: torch.Tensor, flat_expert_indices: torch.Tensor, flat_expert_weights: torch.Tensor):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount(minlength=len(self.experts)).cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        for expert_id, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[expert_id]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)
        return expert_cache


class MiniMindEncoderLayer(nn.Module):
    def __init__(self, config: MiniMindABConfig):
        super().__init__()
        self.self_attn = SelfAttention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states: torch.Tensor, position_embeddings, attention_mask: Optional[torch.Tensor], valid_mask: Optional[torch.Tensor]):
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            attention_mask=attention_mask,
            is_causal=False,
        )
        hidden_states = hidden_states + residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return _apply_valid_mask(hidden_states, valid_mask)


class MiniMindDecoderLayer(nn.Module):
    def __init__(self, config: MiniMindABConfig):
        super().__init__()
        self.self_attn = SelfAttention(config)
        self.cross_attn = CrossAttention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_self_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_cross_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
        self.decoder_causal = config.decoder_causal

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: torch.Tensor,
        position_embeddings,
        self_attention_mask: Optional[torch.Tensor],
        cross_attention_mask: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor],
    ):
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            attention_mask=self_attention_mask,
            is_causal=self.decoder_causal,
        )
        hidden_states = hidden_states + residual
        hidden_states = _apply_valid_mask(hidden_states, valid_mask)

        residual = hidden_states
        hidden_states = self.cross_attn(
            self.post_self_layernorm(hidden_states),
            memory,
            attention_mask=cross_attention_mask,
        )
        hidden_states = hidden_states + residual
        hidden_states = _apply_valid_mask(hidden_states, valid_mask)

        hidden_states = hidden_states + self.mlp(self.post_cross_layernorm(hidden_states))
        return _apply_valid_mask(hidden_states, valid_mask)


class MiniMindABTransformer(nn.Module):
    def __init__(self, config: Optional[MiniMindABConfig] = None):
        super().__init__()
        self.config = config or MiniMindABConfig()
        self.input_proj = nn.Linear(self.config.input_dim, self.config.hidden_size)
        self.fragment_embed = nn.Embedding(self.config.num_fragment_types + 1, self.config.hidden_size, padding_idx=0)
        self.input_dropout = nn.Dropout(self.config.dropout)
        self.encoder_layers = nn.ModuleList(
            [MiniMindEncoderLayer(self.config) for _ in range(self.config.num_hidden_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [MiniMindDecoderLayer(self.config) for _ in range(self.config.num_hidden_layers)]
        )
        self.encoder_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.decoder_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.output_layer_ab = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.output_layer_ag = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=self.config.hidden_size // self.config.num_attention_heads,
            end=self.config.max_position_embeddings,
            rope_base=self.config.rope_theta,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        mask_res: Optional[torch.Tensor] = None,
        fragment_type: Optional[torch.Tensor] = None,
        res_feat: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        aa_labels: Optional[torch.Tensor] = None,
        generate_mask: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ):
        if res_feat is None or fragment_type is None:
            raise ValueError("res_feat and fragment_type are required.")

        if mask_res is None:
            mask_res = torch.ones(res_feat.shape[:2], dtype=torch.bool, device=res_feat.device)
        else:
            mask_res = mask_res.bool()

        fragment_type = fragment_type.long()
        valid_types = fragment_type.clamp(min=0, max=self.config.num_fragment_types)
        hidden_states = self.input_proj(res_feat) + self.fragment_embed(valid_types)
        hidden_states = self.input_dropout(hidden_states)

        heavy_mask = (fragment_type == 1) & mask_res
        light_mask = (fragment_type == 2) & mask_res
        antigen_mask = (fragment_type == 3) & mask_res
        antibody_mask = heavy_mask | light_mask

        ab_feat_pad, ab_mask, ag_feat_pad, ag_mask = extract_and_pad(hidden_states, antibody_mask, antigen_mask)
        enc_mask, dec_self_mask, cross_mask = make_masks_from_pads(ab_mask, ag_mask)

        ag_pos = self._get_position_embeddings(ag_feat_pad.size(1), ag_feat_pad.device)
        memory = ablate_invalid_tokens(ag_feat_pad, ag_mask)
        for layer in self.encoder_layers:
            memory = layer(memory, ag_pos, enc_mask, ag_mask)
        memory = _apply_valid_mask(self.encoder_norm(memory), ag_mask)

        ab_pos = self._get_position_embeddings(ab_feat_pad.size(1), ab_feat_pad.device)
        output = ablate_invalid_tokens(ab_feat_pad, ab_mask)
        for layer in self.decoder_layers:
            output = layer(output, memory, ab_pos, dec_self_mask, cross_mask, ab_mask)
        output = _apply_valid_mask(self.decoder_norm(output), ab_mask)

        ab_logits = self.output_layer_ab(output)
        ag_logits = self.output_layer_ag(memory)
        merged_output = merge_back(
            ab_out=ab_logits,
            ag_out=ag_logits,
            AB_bool=antibody_mask,
            A_bool=antigen_mask,
            fill_value=0.0,
        )

        aux_loss = merged_output.new_zeros(())
        for layer in list(self.encoder_layers) + list(self.decoder_layers):
            if isinstance(layer.mlp, MOEFeedForward):
                aux_loss = aux_loss + layer.mlp.aux_loss.to(aux_loss.device)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                merged_output[mask_res],
                labels[mask_res],
                ignore_index=-100,
            )

        if return_dict:
            return {
                "loss": loss,
                "logits": merged_output,
                "aux_loss": aux_loss,
                "ab_mask": antibody_mask,
                "ag_mask": antigen_mask,
            }

        if labels is not None:
            return merged_output, loss, aux_loss
        return merged_output

    def _get_position_embeddings(self, seq_len: int, device: torch.device):
        if seq_len > self.freqs_cos.size(0):
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_position_embeddings={self.freqs_cos.size(0)}"
            )
        return (
            self.freqs_cos[:seq_len].to(device),
            self.freqs_sin[:seq_len].to(device),
        )


class ABTransformer(MiniMindABTransformer):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        N: int,
        nheads: int,
        dropout: float = 0.0,
        src_input_dim: Optional[int] = None,
        tgt_input_dim: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,
        max_position_embeddings: int = 4096,
        use_moe: bool = False,
        **kwargs,
    ):
        if src_input_dim is None and tgt_input_dim is None:
            raise ValueError("At least one of src_input_dim or tgt_input_dim must be provided.")
        if src_input_dim is not None and tgt_input_dim is not None and src_input_dim != tgt_input_dim:
            raise ValueError("src_input_dim and tgt_input_dim must match for shared MiniMind input projection.")

        input_dim = src_input_dim if src_input_dim is not None else tgt_input_dim
        config = MiniMindABConfig(
            input_dim=input_dim,
            vocab_size=vocab_size,
            hidden_size=d_model,
            num_hidden_layers=N,
            num_attention_heads=nheads,
            num_key_value_heads=num_key_value_heads if num_key_value_heads is not None else max(1, nheads // 4),
            max_position_embeddings=max_position_embeddings,
            dropout=dropout,
            use_moe=use_moe,
            **kwargs,
        )
        super().__init__(config)


Transformer = ABTransformer


def _prepare_attention_mask(
    mask: Optional[torch.Tensor],
    batch_size: int,
    seq_q: int,
    seq_k: int,
    device: torch.device,
):
    if mask is None:
        return None
    if mask.dim() == 2:
        if mask.size(1) == seq_k:
            mask = mask[:, None, None, :]
        elif mask.size(1) == seq_q:
            mask = mask[:, None, :, None].expand(-1, 1, -1, seq_k)
        else:
            raise ValueError(f"Unsupported 2D mask shape: {mask.shape}")
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    elif mask.dim() != 4:
        raise ValueError(f"Unsupported mask rank: {mask.dim()}")
    return mask.to(device=device, dtype=torch.bool)


def _apply_valid_mask(x: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if valid_mask is None:
        return x
    return x * valid_mask.unsqueeze(-1).type_as(x)


def ablate_invalid_tokens(x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    return _apply_valid_mask(x, valid_mask)


def extract_and_pad(res_feat: torch.Tensor, AB_bool: torch.Tensor, A_bool: torch.Tensor):
    batch_size, _, hidden_dim = res_feat.shape
    ab_list = []
    ag_list = []
    for batch_idx in range(batch_size):
        ab_list.append(res_feat[batch_idx][AB_bool[batch_idx]])
        ag_list.append(res_feat[batch_idx][A_bool[batch_idx]])

    ab_max_len = max((x.size(0) for x in ab_list), default=0)
    ag_max_len = max((x.size(0) for x in ag_list), default=0)
    ab_max_len = max(ab_max_len, 1)
    ag_max_len = max(ag_max_len, 1)

    ab_pad = torch.zeros(batch_size, ab_max_len, hidden_dim, device=res_feat.device, dtype=res_feat.dtype)
    ag_pad = torch.zeros(batch_size, ag_max_len, hidden_dim, device=res_feat.device, dtype=res_feat.dtype)
    ab_mask = torch.zeros(batch_size, ab_max_len, dtype=torch.bool, device=res_feat.device)
    ag_mask = torch.zeros(batch_size, ag_max_len, dtype=torch.bool, device=res_feat.device)

    for batch_idx in range(batch_size):
        ab = ab_list[batch_idx]
        ag = ag_list[batch_idx]
        if ab.size(0) > 0:
            ab_pad[batch_idx, :ab.size(0)] = ab
            ab_mask[batch_idx, :ab.size(0)] = True
        if ag.size(0) > 0:
            ag_pad[batch_idx, :ag.size(0)] = ag
            ag_mask[batch_idx, :ag.size(0)] = True
    return ab_pad, ab_mask, ag_pad, ag_mask


def make_masks_from_pads(ab_mask: torch.Tensor, ag_mask: torch.Tensor):
    enc_mask = ag_mask.unsqueeze(1).unsqueeze(2)
    dec_self_mask = ab_mask.unsqueeze(1).unsqueeze(2) & ab_mask.unsqueeze(1).unsqueeze(-1)
    cross_mask = (ab_mask.unsqueeze(-1) & ag_mask.unsqueeze(1)).unsqueeze(1)
    return enc_mask, dec_self_mask, cross_mask


def merge_back(
    ab_out: torch.Tensor,
    ag_out: torch.Tensor,
    AB_bool: torch.Tensor,
    A_bool: torch.Tensor,
    fill_value: float = 0.0,
):
    if AB_bool.shape != A_bool.shape:
        raise ValueError("AB_bool and A_bool must have the same shape.")
    batch_size, seq_len = AB_bool.shape
    if ab_out is not None:
        output_dim = ab_out.shape[-1]
        device = ab_out.device
        dtype = ab_out.dtype
    elif ag_out is not None:
        output_dim = ag_out.shape[-1]
        device = ag_out.device
        dtype = ag_out.dtype
    else:
        raise ValueError("ab_out and ag_out cannot both be None.")

    merged = torch.full((batch_size, seq_len, output_dim), float(fill_value), device=device, dtype=dtype)
    for batch_idx in range(batch_size):
        num_ab = int(AB_bool[batch_idx].sum().item())
        num_ag = int(A_bool[batch_idx].sum().item())
        if num_ab > 0:
            merged[batch_idx, AB_bool[batch_idx]] = ab_out[batch_idx, :num_ab]
        if num_ag > 0:
            merged[batch_idx, A_bool[batch_idx]] = ag_out[batch_idx, :num_ag]
    return merged


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ABTransformer(
        vocab_size=32,
        d_model=64,
        N=2,
        nheads=4,
        dropout=0.0,
        src_input_dim=16,
        tgt_input_dim=16,
        flash_attn=False,
    ).to(device)

    mask_res = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0],
        ],
        dtype=torch.bool,
        device=device,
    )
    fragment_type = torch.tensor(
        [
            [1, 2, 3, 3, 0, 0],
            [1, 3, 2, 0, 0, 0],
        ],
        dtype=torch.long,
        device=device,
    )
    res_feat = torch.randn(2, 6, 16, device=device)

    logits = model(mask_res=mask_res, fragment_type=fragment_type, res_feat=res_feat)
    print("logits shape:", logits.shape)

    labels = torch.randint(0, 32, (2, 6), device=device)
    labels = labels.masked_fill(~mask_res, -100)
    output = model(
        mask_res=mask_res,
        fragment_type=fragment_type,
        res_feat=res_feat,
        labels=labels,
        return_dict=True,
    )
    print("loss:", None if output["loss"] is None else float(output["loss"].detach().cpu()))
    print("aux_loss:", float(output["aux_loss"].detach().cpu()))
