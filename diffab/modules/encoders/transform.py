import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
from typing import Optional


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()

        # 创建位置矩阵 (max_len, 1)
        position = torch.arange(max_len).unsqueeze(1)

        # 创建 div_term (d_model/2,)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        # 计算 PE
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维

        # shape → (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # 不作为参数，随模型保存
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        # 返回加上位置编码的结果
        return x + self.pe[:, :x.size(1)]

class MutiheadAttention(nn.Module):
    def __init__(self, d_model, nheads, dropout=0.1):
        super().__init__()
        self.d_model= d_model
        self.nheads = nheads
        assert d_model % nheads == 0, "d_model must be divisible by nheads"
        self.d_k = d_model // nheads
        self.linear_Q=nn.Linear(d_model, d_model)
        self.linear_K=nn.Linear(d_model, d_model)
        self.linear_V=nn.Linear(d_model, d_model)
        self.dropout=nn.Dropout(dropout)
        self.linear_out=nn.Linear(d_model, d_model)

    def attention(self, Q, K, V, mask=None):
        # Q,K,V: (batch, nheads, seq_len, d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, nheads, seq_q, seq_k)

        if mask is not None:
            # Allow mask shapes:
            #   (batch, seq_k)            -> padding mask (key positions)
            #   (batch, seq_q, seq_k)
            #   (batch, 1, seq_q, seq_k) or (batch, nheads, seq_q, seq_k)
            if mask.dim() == 2:  # (batch, seq_k)
                mask = mask.unsqueeze(1).unsqueeze(1)  # -> (batch,1,1,seq_k)
            elif mask.dim() == 3:  # (batch, seq_q, seq_k)
                mask = mask.unsqueeze(1)  # -> (batch,1,seq_q,seq_k)
            # else assume already (batch,1 or nheads, seq_q, seq_k)

            # broadcast if needed
            if mask.size(1) == 1 and mask.size(1) != scores.size(1):
                mask = mask.expand(-1, scores.size(1), -1, -1)

            scores = scores.masked_fill(mask == 0, float('-1e9'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, V)  # (batch, nheads, seq_q, d_k)
        return output, attn

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, _ = q.size()
        _,seq_len_k,_=k.size()
        _,seq_len_v,_=v.size()
        Q = self.linear_Q(q).view(batch_size, seq_len, self.nheads, self.d_k).transpose(1, 2)
        K = self.linear_K(k).view(batch_size, seq_len_k, self.nheads, self.d_k).transpose(1, 2)
        V = self.linear_V(v).view(batch_size, seq_len_v, self.nheads, self.d_k).transpose(1, 2)

        output, attn = self.attention(Q, K, V, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.linear_out(output)
        return output, attn

class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=512, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class NormLayer(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x - mean) / (std + self.eps)
        return self.alpha * x + self.bias

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nheads, dropout=0.1):
        super().__init__()
        self.norm1 = NormLayer(d_model)
        self.norm2 = NormLayer(d_model)
        self.attention = MutiheadAttention(d_model, nheads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff=512, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.attention(x2, x2, x2, mask)[0])
        x2 = self.norm2(x)
        x = x + self.dropout2(self.feed_forward(x2))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size=None, d_model=512, N=6, nheads=8, dropout=0.1, input_dim=None):
        """
        If input_dim is provided (e.g. 128), the encoder accepts continuous features of shape (B, L, input_dim).
        Otherwise it expects token ids (LongTensor) and uses nn.Embedding(vocab_size, d_model).
        """
        super().__init__()
        self.N = N
        self.d_model = d_model
        self.input_dim = input_dim
        if input_dim is None:
            assert vocab_size is not None, "vocab_size must be provided if input_dim is None"
            self.embed = nn.Embedding(vocab_size, d_model)
            self.input_proj = None
        else:
            # project input features -> d_model
            self.input_proj = nn.Linear(input_dim, d_model)
            self.embed = None

        self.pos_embed = PositionEmbedding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, nheads, dropout) for _ in range(N)])
        self.norm = NormLayer(d_model)

    def forward(self, x, mask=None):
        # x can be LongTensor token ids (B,L) or float features (B,L,input_dim)
        if self.embed is not None:
            x = self.embed(x) * math.sqrt(self.embed.embedding_dim)
        else:
            x = self.input_proj(x)  # (B,L,d_model)
        x = self.pos_embed(x)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nheads, dropout=0.1):
        super().__init__()
        self.norm1 = NormLayer(d_model)
        self.norm2 = NormLayer(d_model)
        self.norm3 = NormLayer(d_model)
        self.attention1 = MutiheadAttention(d_model, nheads, dropout)
        self.attention2 = MutiheadAttention(d_model, nheads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff=512, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.attention1(x2, x2, x2, tgt_mask)[0])
        x2 = self.norm2(x)
        x = x + self.dropout2(self.attention2(x2, memory, memory, src_mask)[0])
        x2 = self.norm3(x)
        x = x + self.dropout3(self.feed_forward(x2))
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size=None, d_model=512, N=6, nheads=8, dropout=0.1, input_dim=None):
        super().__init__()
        self.N = N
        self.d_model = d_model
        self.input_dim = input_dim
        if input_dim is None:
            assert vocab_size is not None, "vocab_size must be provided if input_dim is None"
            self.embed = nn.Embedding(vocab_size, d_model)
            self.input_proj = None
        else:
            self.input_proj = nn.Linear(input_dim, d_model)
            self.embed = None

        self.pos_embed = PositionEmbedding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, nheads, dropout) for _ in range(N)])
        self.norm = NormLayer(d_model)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        if self.embed is not None:
            x = self.embed(x) * math.sqrt(self.embed.embedding_dim)
        else:
            # print("x shape before input_proj:", x.shape)
            x = self.input_proj(x)
        x = self.pos_embed(x)

        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)

        x = self.norm(x)
        return x

class ABTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, N, nheads, dropout=0.1, src_input_dim=None, tgt_input_dim=None):
        super().__init__()
        # encoder/decoder accept either token ids or continuous features (if input_dim provided)
        self.encoder = Encoder(vocab_size=vocab_size, d_model=d_model, N=N, nheads=nheads, dropout=dropout, input_dim=src_input_dim)
        self.decoder = Decoder(vocab_size=vocab_size, d_model=d_model, N=N, nheads=nheads, dropout=dropout, input_dim=tgt_input_dim)
        self.output_layer_ab=nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, vocab_size)
        )
        self.output_layer_ag=nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, vocab_size)
        )

    def forward(
        self, 
        mask_res=None,
        fragment_type=None,
        res_feat=None
    ):
        """
        mask_res: (N, L)    1=真实残基, 0=padding
        fragment_type: (N, L)   1=重链, 2=轻链, 3=抗原链
        res_feat: (N, L, res_dim)
        逻辑: 重链+轻链 做 Q，抗原 做 K/V
        """

        N, L, _ = res_feat.shape

        # -----------------------------
        # 1. 生成 Boolean masks
        # -----------------------------
        H_bool = (fragment_type == 1)   # heavy
        L_bool = (fragment_type == 2)   # light
        A_bool = (fragment_type == 3)   # antigen

        AB_bool = H_bool | L_bool
        ab_feat_pad, ab_mask, ag_feat_pad, ag_mask = extract_and_pad(res_feat, AB_bool, A_bool)
        # print("ab_feat_pad:", ab_feat_pad.shape)
        # print("ag_feat_pad:", ag_feat_pad.shape)

        # 生成三类 mask
        enc_mask, dec_self_mask, cross_mask = make_masks_from_pads(ab_mask, ag_mask)

        # 如果需要因果 mask（decoder）可加入：
        causal = torch.tril(torch.ones(dec_self_mask.size(-1), dec_self_mask.size(-1), dtype=torch.bool, device=res_feat.device))
        dec_self_mask = dec_self_mask & causal.unsqueeze(0).unsqueeze(1)  # (N,1,tgt_len,tgt_len)

        # encoder 只处理抗原（ag_feat_pad）
        memory = self.encoder(ag_feat_pad, mask=enc_mask)

        # decoder 只处理抗体：传入 dec_self_mask（self-attn） 和 cross_mask（cross-attn）
        output = self.decoder(
            ab_feat_pad,
            memory,
            src_mask=cross_mask,   # cross-attn uses this mask (N,1,tgt_len,src_len)
            tgt_mask=dec_self_mask # self-attn mask (N,1,tgt_len,tgt_len)
        )

        ab_out = self.output_layer_ab(output)
        ag_out=self.output_layer_ag(memory)
        merged_output = merge_back(
            ab_out=ab_out,
            ag_out=ag_out,
            AB_bool=AB_bool,
            A_bool=A_bool,
            fill_value=0.0
        )
        return merged_output





def extract_and_pad(res_feat, AB_bool, A_bool):
    """
    返回：
      ab_feat_pad: (N, L_ab_max, C)
      ab_mask:     (N, L_ab_max)  True=有效
      ag_feat_pad: (N, L_ag_max, C)
      ag_mask:     (N, L_ag_max)
    处理 edge case：若某一类在整个 batch 中都没有（长度为0），会用长度1的 dummy 填充并把 mask 设为 False
    """
    N, L, C = res_feat.shape

    ab_list = []
    ag_list = []

    for i in range(N):
        ab_list.append(res_feat[i][AB_bool[i]])   # (len_i_ab, C)
        ag_list.append(res_feat[i][A_bool[i]])   # (len_i_ag, C)

    L_ab_max = max((x.size(0) for x in ab_list), default=0)
    L_ag_max = max((x.size(0) for x in ag_list), default=0)

    # protect against all-zero case: make minimal length 1 to avoid (N,0,C) tensors
    if L_ab_max == 0:
        L_ab_max = 1
    if L_ag_max == 0:
        L_ag_max = 1

    ab_pad = torch.zeros(N, L_ab_max, C, device=res_feat.device, dtype=res_feat.dtype)
    ag_pad = torch.zeros(N, L_ag_max, C, device=res_feat.device, dtype=res_feat.dtype)

    ab_mask = torch.zeros(N, L_ab_max, dtype=torch.bool, device=res_feat.device)
    ag_mask = torch.zeros(N, L_ag_max, dtype=torch.bool, device=res_feat.device)

    for i in range(N):
        ab = ab_list[i]
        ag = ag_list[i]

        if ab.size(0) > 0:
            ab_pad[i, :ab.size(0)] = ab
            ab_mask[i, :ab.size(0)] = True
        # else leave zeros and mask False (dummy)

        if ag.size(0) > 0:
            ag_pad[i, :ag.size(0)] = ag
            ag_mask[i, :ag.size(0)] = True

    return ab_pad, ab_mask, ag_pad, ag_mask


def make_masks_from_pads(ab_mask, ag_mask):
    """
    Input:
      ab_mask: (N, tgt_len)  True=有效
      ag_mask: (N, src_len)  True=有效
    Returns:
      enc_mask:      (N, 1, 1, src_len)    用于 encoder self-attn / cross-attn key masking
      dec_self_mask: (N, 1, tgt_len, tgt_len)  用于 decoder self-attn (可与 causal 合并)
      cross_mask:    (N, 1, tgt_len, src_len)  用于 decoder cross-attn (query=tgt, key=src)
    语义：True = allowed（有效）；attention 中请使用 masked_fill(~mask, -inf)
    """
    N, tgt_len = ab_mask.shape
    _, src_len = ag_mask.shape
    device = ab_mask.device

    # encoder mask: (N,1,1,src_len) - True 表示 key 可用（非 padding）
    enc_mask = ag_mask.unsqueeze(1).unsqueeze(2)  # (N,1,1,src_len)

    # decoder self-attn mask: (N,1,tgt_len,tgt_len)
    dec_self = ab_mask.unsqueeze(1).unsqueeze(2)  # (N,1,1,tgt_len)
    dec_self = dec_self & dec_self.transpose(-1, -2)  # (N,1,tgt_len,tgt_len)

    # cross-attn mask: (N,1,tgt_len,src_len)
    # allow attention only where both query (tgt pos) is valid and key (src pos) is valid
    cross = ab_mask.unsqueeze(-1) & ag_mask.unsqueeze(1)  # (N, tgt_len, src_len)
    cross = cross.unsqueeze(1)  # (N,1,tgt_len,src_len)

    return enc_mask, dec_self, cross

def merge_back(
    ab_out: torch.Tensor,
    ag_out: torch.Tensor,
    AB_bool: torch.Tensor,
    A_bool: torch.Tensor,
    fill_value: float = 0.0
) -> torch.Tensor:
    """
    将按类处理后的特征拼回原始 (N, L, D) 布局。

    输入:
      ab_out: (N, L_ab_max, D)   抗体处理后的输出（可能是 decoder 输出）
      ag_out: (N, L_ag_max, D)   抗原处理后的输出（可能是 encoder 输出）
      AB_bool: (N, L) bool       抗体位置掩码（True 表示该位置属于抗体）
      A_bool:  (N, L) bool       抗原位置掩码（True 表示该位置属于抗原）
      fill_value: 数值，填充 padding 位置（默认 0.0）

    返回:
      merged: (N, L, D)          恢复回原始序列长度 L 的张量
    """
    assert AB_bool.shape == A_bool.shape, "AB_bool 和 A_bool 形状必须相同 (N, L)"
    N, L = AB_bool.shape

    # 推断输出维度 D（优先使用 ab_out）
    D = None
    if ab_out is not None:
        D = ab_out.shape[2]
        device = ab_out.device
        dtype = ab_out.dtype
    elif ag_out is not None:
        D = ag_out.shape[2]
        device = ag_out.device
        dtype = ag_out.dtype
    else:
        raise ValueError("ab_out 和 ag_out 不能同时为 None")

    # 准备结果张量，默认填充 fill_value
    merged = torch.full((N, L, D), float(fill_value), device=device, dtype=dtype)

    # 按样本逐个 scatter 回去（N 一般不大，循环开销可接受）
    for i in range(N):
        # 抗体
        n_ab = int(AB_bool[i].sum().item())
        if n_ab > 0:
            # ab_out 的前 n_ab 项对应该样本的抗体位置（与 extract_and_pad 保持一致）
            merged[i, AB_bool[i]] = ab_out[i, :n_ab]

        # 抗原
        n_ag = int(A_bool[i].sum().item())
        if n_ag > 0:
            merged[i, A_bool[i]] = ag_out[i, :n_ag]

    return merged




if __name__ == "__main__":




    batch = 4
    src_len = 6
    tgt_len = 6
    feat_dim = 128
    vocab_size = 100
    d_model = 512

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====== 1.  (batch, seq_len, 128) ======
    src = torch.randn(batch, src_len, feat_dim).to(device)
    tgt = torch.randn(batch, tgt_len, feat_dim).to(device)

    # ====== 2. padding mask  (batch, seq_len) ======

    src_padding_mask = torch.tensor([
        [1, 1, 1, 1, 0, 0],  
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0],
    ]).to(device)


    tgt_padding_mask = torch.tensor([
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0],
    ]).to(device)


    # ====== 3. decoder causal mask (seq_len, seq_len) ======
    def generate_causal_mask(L):
        mask = torch.tril(torch.ones(L, L)).to(device) 
        return mask.unsqueeze(0).unsqueeze(1)  # (L, L)

    tgt_causal_mask = generate_causal_mask(tgt_len)



    model = Transformer(
        vocab_size=feat_dim,
        d_model=d_model,
        N=3,
        nheads=8,
        dropout=0.1,
        src_input_dim=128,   
        tgt_input_dim=128
    ).to(device)

    # ====== 5. forward ======
    output = model(
        src,
        tgt,
        src_mask=None,
        tgt_mask=tgt_causal_mask
    )

    print("output:", output.shape)

