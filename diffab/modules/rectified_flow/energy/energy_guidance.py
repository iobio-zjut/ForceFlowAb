import torch
from madrax.ForceField import ForceField
# from madrax.ForceFieldLZH import ForceField
from madrax import dataStructures
# -----------------------------
# 1) lambda 调度（推理阶段用）
# -----------------------------
def lambda_schedule(t, T, max_lambda=0.2, p=2.0, tail_drop=0.1):
    """
    t: 当前时间步（整数，通常从 T-1 递减到 0）
    T: 总步数
    返回 λ_t：前期小、逐步增大，最后 tail_drop*100% 再线性降到 0，避免后期“猛拽”。
    """
    # 归一化到 0->1，越靠后越大
    x = 1.0 - (t / float(T))
    lam = max_lambda * (x ** p)
    # 最后 tail_drop 比例（默认 10%）线性降回 0
    if x > (1.0 - tail_drop):
        lam *= (1.0 - (x - (1.0 - tail_drop)) / tail_drop)
    return float(lam)

# --------------------------------------------------------
# 4) 主类：读取 16 个 pdb → MadraX → 只对 A/B 链 CDR 区计算能量 & 返回 Cα 梯度
# --------------------------------------------------------
class EnergyGuidance:
    def __init__(self, device="cuda:1", use_weighted_backbone_terms=True):
        """
        use_weighted_backbone_terms=True: 仅保留对骨架可信的能量项（权重见 _energy_weights）
        False: 直接对 11 项等权求和
        """
        self.device = device
        self.ff = ForceField(device=device)
        self.use_weighted = use_weighted_backbone_terms
        self.energy_weights = self._energy_weights(device) if use_weighted_backbone_terms else None
        self.repulsion_weight = 10.0
        self._info_tensor_cache = {}
        self._pair_mask_cache = {}
    
    @staticmethod
    def _energy_weights(device):
        # 索引对应关系：
        # 0:SS (二级结构)
        # 1:HB (氢键) -> 保留，促进形成螺旋/折叠
        # 2:Elec (静电) -> 关掉，侧链电荷未定义
        # 3:VDW_Clash (严重碰撞) -> 关掉！防止因为没有侧链而爆炸
        # 4:Polar (极性) -> 关掉
        # 5:Hydrophobic (疏水) -> 关掉
        # 6:VDW (范德华) -> 关掉
        # 7:BB_Ent (骨架熵/Rama图) -> 保留，保证骨架角度自然
        # 8:SC_Ent (侧链熵) -> 关掉
        # 9:Peptide (肽键约束) -> 保留，保证连接处不扭曲
        # 10:Rotamer (侧链构象) -> 关掉
        
        # 修改后的权重：只留 1, 7, 9
        w = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.1, 0.0]
        
        # 注意：BB_Ent 有时数值较大，可以给 0.5 或 0.1，具体看梯度大小
        return torch.tensor(w, device=device, dtype=torch.float32)

    @staticmethod
    def _atnames_cache_key(atnames):
        if len(atnames) > 0 and isinstance(atnames[0], list):
            atnames = atnames[0]
        return tuple(str(name) for name in atnames)

    def _get_info_tuple(self, atnames):
        cache_key = self._atnames_cache_key(atnames)
        info_tuple = self._info_tensor_cache.get(cache_key)
        if info_tuple is None:
            info_tuple = dataStructures.create_info_tensors(atnames, device=self.device)
            self._info_tensor_cache[cache_key] = info_tuple
        return info_tuple

    def _get_non_adjacent_pair_mask(self, num_ca, device):
        cache_key = (str(device), int(num_ca))
        pair_mask = self._pair_mask_cache.get(cache_key)
        if pair_mask is None:
            i_idx, j_idx = torch.triu_indices(num_ca, num_ca, offset=1, device=device)
            pair_mask = (j_idx - i_idx) > 1
            self._pair_mask_cache[cache_key] = pair_mask
        return pair_mask

    def _calc_ca_repulsion(self, coords, mask_atom, min_dist=3.5):
        """
        修正版 v2：自动处理 coords 和 mask_atom 长度不一致的问题
        """
        L = min(coords.shape[1], mask_atom.shape[1])
        ca_bool = mask_atom[0, :L, 0].bool()
        num_ca = int(ca_bool.sum().item())
        if num_ca < 2:
            return coords.new_zeros(())

        ca_coords = coords[0, :L, :][ca_bool]
        pair_mask = self._get_non_adjacent_pair_mask(num_ca, coords.device)
        if not torch.any(pair_mask):
            return coords.new_zeros(())

        dists = torch.pdist(ca_coords, p=2) + 1e-8
        violation = torch.nn.functional.relu(min_dist - dists[pair_mask])
        return 0.5 * torch.sum(violation.square())


    def compute_grad_batch(self, pdb_name, cdr_mask, coords, atnames, mask_atom, return_meta=False):
        info_tuple = self._get_info_tuple(atnames)
        cdr_mask = cdr_mask.to(self.device, dtype=torch.float32)
        mask_atom = mask_atom.to(self.device)

        with torch.enable_grad():
            coords = coords.to(self.device).clone().detach().requires_grad_(True)
            energy = self.ff(coords, info_tuple)

            if self.use_weighted:
                E_used = torch.tensordot(energy, self.energy_weights, dims=([-1], [0]))
            else:
                E_used = energy.sum(dim=-1)

            E_madrax = torch.sum(E_used * cdr_mask)
            E_repulsion = self._calc_ca_repulsion(coords, mask_atom, min_dist=3.5)
            E_total = E_madrax + self.repulsion_weight * E_repulsion
            grad_all = torch.autograd.grad(E_total, coords, retain_graph=False, create_graph=False)[0]
            total_energy = float(E_total.detach().item())

        L = min(grad_all.shape[1], mask_atom.shape[1])
        grad_all = grad_all[:, :L, :]
        ca_bool = mask_atom[0, :L, 0].bool()
        grad_ca = grad_all[:, ca_bool, :]
        coord_ca = coords[:, :L, :][:, ca_bool, :]  # 410

        if grad_ca.numel() > 0:
            gnorm = grad_ca.norm(dim=-1, keepdim=True) + 1e-8
            force_ca = grad_ca * (torch.clamp(gnorm, 0, 1) / gnorm)  # 410

            ca_center = coord_ca.mean(dim=1, keepdim=True)  # 410
            lever = coord_ca - ca_center  # 410
            torque_ca = torch.cross(lever, force_ca, dim=-1)  # 410
            tnorm = torque_ca.norm(dim=-1, keepdim=True) + 1e-8  # 410
            torque_ca = torque_ca * (torch.clamp(tnorm, 0, 1) / tnorm)  # 410
        else:
            force_ca = grad_ca  # 410
            torque_ca = grad_ca  # 410

        guidance = {"force": force_ca.detach(), "torque": torque_ca.detach()}  # 410
        meta = None
        if return_meta:
            meta = {
                "pdb_name": pdb_name,
                "E_total": total_energy,
            }

        return guidance, meta  # 410


