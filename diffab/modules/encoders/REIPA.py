import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from diffab.modules.common.geometry import global_to_local, local_to_global, normalize_vector, construct_3d_basis, angstrom_to_nm
from diffab.modules.common.layers import mask_zero, LayerNorm
from diffab.utils.protein.constants import BBHeavyAtom
from diffab.modules.encoders.IPA import InvariantPointAttention   #LZH
from diffab.modules.encoders.src.net.ipa import TranslationIPA #LZH


def _alpha_from_logits(logits, mask, inf=1e5):
    """
    Args:
        logits: Logit matrices, (N, L_i, L_j, num_heads).
        mask:   Masks, (N, L).
    Returns:
        alpha:  Attention weights.
    """
    N, L, _, _ = logits.size()
    mask_row = mask.view(N, L, 1, 1).expand_as(logits)  # (N, L, *, *)
    mask_pair = mask_row * mask_row.permute(0, 2, 1, 3)  # (N, L, L, *)

    logits = torch.where(mask_pair, logits, logits - inf)
    alpha = torch.softmax(logits, dim=2)  # (N, L, L, num_heads)
    alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))
    return alpha


def _heads(x, n_heads, n_ch):
    """
    Args:
        x:  (..., num_heads * num_channels)
    Returns:
        (..., num_heads, num_channels)
    """
    s = list(x.size())[:-1] + [n_heads, n_ch]
    return x.view(*s)



def rotation_matrix_to_quaternion(R, eps=1e-8):
    """
    Convert 3x3 rotation matrix to 4d quaternion representation.
    Args:
        R: (..., 3, 3) rotation matrices
        eps: small value to avoid sqrt(0) or division by zero
    Returns:
        q: (..., 4) quaternion in (w, x, y, z) order
    """
    # 确保输入是 contiguous 的，避免 view/reshape 出错
    if not R.is_contiguous():
        R = R.contiguous()

    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    
    batch_size = R.shape[0]
    q = torch.zeros((batch_size, 4), device=R.device, dtype=R.dtype)

    # 提取对角线元素
    R00 = R[:, 0, 0]
    R11 = R[:, 1, 1]
    R22 = R[:, 2, 2]
    trace = R00 + R11 + R22

    # Case 1: Trace > 0
    mask_trace = trace > 0
    if mask_trace.any():
        # 加 eps 防止浮点误差导致的负数
        t = torch.sqrt(torch.clamp(trace[mask_trace] + 1.0, min=eps)) * 2.0
        q[mask_trace, 0] = 0.25 * t
        q[mask_trace, 1] = (R[mask_trace, 2, 1] - R[mask_trace, 1, 2]) / t
        q[mask_trace, 2] = (R[mask_trace, 0, 2] - R[mask_trace, 2, 0]) / t
        q[mask_trace, 3] = (R[mask_trace, 1, 0] - R[mask_trace, 0, 1]) / t

    # Case 2: R00 is max diagonal
    # 使用与非逻辑，确保互斥
    mask0 = (~mask_trace) & (R00 >= R11) & (R00 >= R22)
    if mask0.any():
        t = torch.sqrt(torch.clamp(1.0 + R00[mask0] - R11[mask0] - R22[mask0], min=eps)) * 2.0
        q[mask0, 0] = (R[mask0, 2, 1] - R[mask0, 1, 2]) / t
        q[mask0, 1] = 0.25 * t
        q[mask0, 2] = (R[mask0, 0, 1] + R[mask0, 1, 0]) / t
        q[mask0, 3] = (R[mask0, 0, 2] + R[mask0, 2, 0]) / t

    # Case 3: R11 is max diagonal
    mask1 = (~mask_trace) & (~mask0) & (R11 >= R22)
    if mask1.any():
        t = torch.sqrt(torch.clamp(1.0 + R11[mask1] - R00[mask1] - R22[mask1], min=eps)) * 2.0
        q[mask1, 0] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / t
        q[mask1, 1] = (R[mask1, 0, 1] + R[mask1, 1, 0]) / t
        q[mask1, 2] = 0.25 * t
        q[mask1, 3] = (R[mask1, 1, 2] + R[mask1, 2, 1]) / t

    # Case 4: R22 is max diagonal
    mask2 = (~mask_trace) & (~mask0) & (~mask1)
    if mask2.any():
        t = torch.sqrt(torch.clamp(1.0 + R22[mask2] - R00[mask2] - R11[mask2], min=eps)) * 2.0
        q[mask2, 0] = (R[mask2, 1, 0] - R[mask2, 0, 1]) / t
        q[mask2, 1] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / t
        q[mask2, 2] = (R[mask2, 1, 2] + R[mask2, 2, 1]) / t
        q[mask2, 3] = 0.25 * t

    # 最后再做一次归一化，保证是单位四元数
    q = torch.nn.functional.normalize(q, dim=-1)
    return q.reshape(batch_shape + (4,))

class GABlock(nn.Module):

    def __init__(self, node_feat_dim, pair_feat_dim, value_dim=32, query_key_dim=32, num_query_points=8,
                 num_value_points=8, num_heads=12, bias=False):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.pair_feat_dim = pair_feat_dim
        self.value_dim = value_dim
        self.query_key_dim = query_key_dim
        self.num_query_points = num_query_points
        self.num_value_points = num_value_points
        self.num_heads = num_heads

        # Node
        self.proj_query = nn.Linear(node_feat_dim, query_key_dim * num_heads, bias=bias)
        self.proj_key = nn.Linear(node_feat_dim, query_key_dim * num_heads, bias=bias)
        self.proj_value = nn.Linear(node_feat_dim, value_dim * num_heads, bias=bias)

        # Pair
        self.proj_pair_bias = nn.Linear(pair_feat_dim, num_heads, bias=bias)

        # Spatial
        self.spatial_coef = nn.Parameter(torch.full([1, 1, 1, self.num_heads], fill_value=np.log(np.exp(1.) - 1.)),
                                         requires_grad=True)
        self.proj_query_point = nn.Linear(node_feat_dim, num_query_points * num_heads * 3, bias=bias)
        self.proj_key_point = nn.Linear(node_feat_dim, num_query_points * num_heads * 3, bias=bias)
        self.proj_value_point = nn.Linear(node_feat_dim, num_value_points * num_heads * 3, bias=bias)

        # Output
        self.out_transform = nn.Linear(
            in_features=(num_heads * pair_feat_dim) + (num_heads * value_dim) + (
                    num_heads * num_value_points * (3 + 3 + 1)),
            out_features=node_feat_dim,
        )

        self.layer_norm_1 = LayerNorm(node_feat_dim)
        self.mlp_transition = nn.Sequential(nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
                                            nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
                                            nn.Linear(node_feat_dim, node_feat_dim))
        self.layer_norm_2 = LayerNorm(node_feat_dim)

    def _node_logits(self, x):
        query_l = _heads(self.proj_query(x), self.num_heads, self.query_key_dim)  # (N, L, n_heads, qk_ch)
        key_l = _heads(self.proj_key(x), self.num_heads, self.query_key_dim)  # (N, L, n_heads, qk_ch)
        logits_node = (query_l.unsqueeze(2) * key_l.unsqueeze(1) *
                       (1 / np.sqrt(self.query_key_dim))).sum(-1)  # (N, L, L, num_heads)
        return logits_node

    def _pair_logits(self, z):
        logits_pair = self.proj_pair_bias(z)
        return logits_pair

    def _spatial_logits(self, R, t, x):
        N, L, _ = t.size()

        # Query
        query_points = _heads(self.proj_query_point(x), self.num_heads * self.num_query_points,
                              3)  # (N, L, n_heads * n_pnts, 3)
        query_points = local_to_global(R, t, query_points)  # Global query coordinates, (N, L, n_heads * n_pnts, 3)
        query_s = query_points.reshape(N, L, self.num_heads, -1)  # (N, L, n_heads, n_pnts*3)

        # Key
        key_points = _heads(self.proj_key_point(x), self.num_heads * self.num_query_points,
                            3)  # (N, L, 3, n_heads * n_pnts)
        key_points = local_to_global(R, t, key_points)  # Global key coordinates, (N, L, n_heads * n_pnts, 3)
        key_s = key_points.reshape(N, L, self.num_heads, -1)  # (N, L, n_heads, n_pnts*3)

        # Q-K Product
        sum_sq_dist = ((query_s.unsqueeze(2) - key_s.unsqueeze(1)) ** 2).sum(-1)  # (N, L, L, n_heads)
        gamma = F.softplus(self.spatial_coef)
        logits_spatial = sum_sq_dist * ((-1 * gamma * np.sqrt(2 / (9 * self.num_query_points)))
                                        / 2)  # (N, L, L, n_heads)
        return logits_spatial

    def _pair_aggregation(self, alpha, z):
        N, L = z.shape[:2]
        feat_p2n = alpha.unsqueeze(-1) * z.unsqueeze(-2)  # (N, L, L, n_heads, C)
        feat_p2n = feat_p2n.sum(dim=2)  # (N, L, n_heads, C)
        return feat_p2n.reshape(N, L, -1)

    def _node_aggregation(self, alpha, x):
        N, L = x.shape[:2]
        value_l = _heads(self.proj_value(x), self.num_heads, self.query_key_dim)  # (N, L, n_heads, v_ch)
        feat_node = alpha.unsqueeze(-1) * value_l.unsqueeze(1)  # (N, L, L, n_heads, *) @ (N, *, L, n_heads, v_ch)
        feat_node = feat_node.sum(dim=2)  # (N, L, n_heads, v_ch)
        return feat_node.reshape(N, L, -1)

    def _spatial_aggregation(self, alpha, R, t, x):
        N, L, _ = t.size()
        value_points = _heads(self.proj_value_point(x), self.num_heads * self.num_value_points,
                              3)  # (N, L, n_heads * n_v_pnts, 3)
        value_points = local_to_global(R, t, value_points.reshape(N, L, self.num_heads, self.num_value_points,
                                                                  3))  # (N, L, n_heads, n_v_pnts, 3)
        aggr_points = alpha.reshape(N, L, L, self.num_heads, 1, 1) * \
                      value_points.unsqueeze(1)  # (N, *, L, n_heads, n_pnts, 3)
        aggr_points = aggr_points.sum(dim=2)  # (N, L, n_heads, n_pnts, 3)

        feat_points = global_to_local(R, t, aggr_points)  # (N, L, n_heads, n_pnts, 3)
        feat_distance = feat_points.norm(dim=-1)  # (N, L, n_heads, n_pnts)
        feat_direction = normalize_vector(feat_points, dim=-1, eps=1e-4)  # (N, L, n_heads, n_pnts, 3)

        feat_spatial = torch.cat([
            feat_points.reshape(N, L, -1),
            feat_distance.reshape(N, L, -1),
            feat_direction.reshape(N, L, -1),
        ], dim=-1)

        return feat_spatial

    def forward(self, R, t, x, z, mask):
        """
        Args:
            R:  Frame basis matrices, (N, L, 3, 3_index).
            t:  Frame external (absolute) coordinates, (N, L, 3).
            x:  Node-wise features, (N, L, F).
            z:  Pair-wise features, (N, L, L, C).
            mask:   Masks, (N, L).
        Returns:
            x': Updated node-wise features, (N, L, F).
        """
        # Attention logits
        logits_node = self._node_logits(x)
        logits_pair = self._pair_logits(z)
        logits_spatial = self._spatial_logits(R, t, x)
        # Summing logits up and apply `softmax`.
        logits_sum = logits_node + logits_pair + logits_spatial
        alpha = _alpha_from_logits(logits_sum * np.sqrt(1 / 3), mask)  # (N, L, L, n_heads)

        # Aggregate features
        feat_p2n = self._pair_aggregation(alpha, z)
        feat_node = self._node_aggregation(alpha, x)
        feat_spatial = self._spatial_aggregation(alpha, R, t, x)

        # Finally
        feat_all = self.out_transform(torch.cat([feat_p2n, feat_node, feat_spatial], dim=-1))  # (N, L, F)
        feat_all = mask_zero(mask.unsqueeze(-1), feat_all)
        x_updated = self.layer_norm_1(x + feat_all)
        x_updated = self.layer_norm_2(x_updated + self.mlp_transition(x_updated))
        return x_updated


class REIPA(nn.Module):

    def __init__(self, node_feat_dim, pair_feat_dim, num_layers):
        super(REIPA, self).__init__()

        self.num_layers = num_layers
        self.translation_ipa = TranslationIPA(c_s=node_feat_dim,
        c_z=pair_feat_dim,
        coordinate_scaling=1.0,
        no_ipa_blocks=self.num_layers,
        skip_embed_size=64,
        transformer_num_heads = 4,
        transformer_num_layers = 2,
        c_hidden = 256,
        no_heads = 8,
        no_qk_points = 8,
        no_v_points = 12,
        dropout = 0.0,)


    def forward(self, R, t, res_feat, pair_feat, mask,mask_generate):   #需要mask_generate，需要rotation_matrix_to_quaternion

        q_t = rotation_matrix_to_quaternion(R)   # (N, L, 4)
        rigids_t = torch.cat([q_t, t], dim=-1)

        batch = {
        "residue_mask": mask,
        "fixed_mask": ~mask_generate,
        "rigids_t": rigids_t,
        }
        output_dict=self.translation_ipa(res_feat,pair_feat,batch)
        return output_dict['embed']

