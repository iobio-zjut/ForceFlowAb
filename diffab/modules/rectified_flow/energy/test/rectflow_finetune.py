import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from tqdm.auto import tqdm
import numpy as np
import logging

from diffab.modules.common.geometry import apply_rotation_to_vector, quaternion_1ijk_to_rotation_matrix, construct_3d_basis_from_e
from diffab.modules.common.so3 import so3vec_to_rotation, rotation_to_so3vec, random_uniform_so3, compute_global_R_from_vec_to_standard,compute_global_R_from_standard_to_vec, rotation_to_quaternion, quaternion_to_rotation_matrix, quaternion_diff, grassmann_product
from diffab.modules.encoders.ga import GAEncoder
from .utils import matrix_to_euler_angles, euler_angles_to_matrix

from diffab.modules.common.layers import clampped_one_hot 

from .trans import AminoacidCategoricalTransition, PositionTransition, QuaternionTransition
from diffab.modules.encoders.REIPA import REIPA
from diffab.modules.encoders.transform import ABTransformer
#
import os
from diffab.modules.common.geometry import reconstruct_backbone_partially   #lll
from diffab.utils.protein.writers import save_pdb
from diffab.utils.data import *
from diffab.modules.rectified_flow.energy.abnum import run_energy_guidance
#

def rotation_matrix_cosine_loss(R_pred, R_true):
    """
    Args:
        R_pred: (*, 3, 3).
        R_true: (*, 3, 3).
    Returns:
        Per-matrix losses, (*, ).
    """
    size = list(R_pred.shape[:-2])
    ncol = R_pred.numel() // 3

    RT_pred = R_pred.transpose(-2, -1).reshape(ncol, 3) # (ncol, 3)
    RT_true = R_true.transpose(-2, -1).reshape(ncol, 3) # (ncol, 3)

    ones = torch.ones([ncol, ], dtype=torch.long, device=R_pred.device)
    loss = F.cosine_embedding_loss(RT_pred, RT_true, ones, reduction='none')  # (ncol*3, )
    loss = loss.reshape(size + [3]).sum(dim=-1)    # (*, )
    return loss


class PredictNet(nn.Module):

    def __init__(self, res_feat_dim, pair_feat_dim, num_layers, encoder_opt={}):
        super().__init__()
        self.current_sequence_embedding = nn.Linear(20, res_feat_dim, bias=True)  # 22 is padding
        self.t_res_embedding = nn.Linear(1, res_feat_dim, bias=True)
        self.t_pair_embedding = nn.Linear(1, pair_feat_dim, bias=True)
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(res_feat_dim * 2, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim),
        )
        # self.encoder = GAEncoder(res_feat_dim, pair_feat_dim, num_layers, **encoder_opt)
        self.reipa=REIPA(res_feat_dim, pair_feat_dim, num_layers) ##use this
        self.abtransformer = ABTransformer(
            vocab_size=res_feat_dim,
            d_model=res_feat_dim*2,
            N=3,
            nheads=8,
            dropout=0.1,
            src_input_dim=res_feat_dim,
            tgt_input_dim=res_feat_dim
        )

        self.seq_net = nn.Sequential(
            nn.Linear(res_feat_dim, res_feat_dim*2), nn.ReLU(),
            nn.Linear(res_feat_dim*2, res_feat_dim*2), nn.ReLU(),
            nn.Linear(res_feat_dim*2, 20),
        )

        self.crd_net = nn.Sequential(
            nn.Linear(res_feat_dim, res_feat_dim*2), nn.ReLU(),
            nn.Linear(res_feat_dim*2, res_feat_dim*2), nn.ReLU(),
            nn.Linear(res_feat_dim*2, 3)
        )
        self.quaternion_net = nn.Sequential(
            nn.Linear(res_feat_dim, res_feat_dim*2), nn.ReLU(),
            nn.Linear(res_feat_dim*2, res_feat_dim*2), nn.ReLU(),
            nn.Linear(res_feat_dim*2, 4)
        )

    def forward(self, R, p, s, t, res_feat, pair_feat, mask_generate, mask_res, fragment_type=None):
        """
        We directly predict the position of all heavy atoms (N, CA, C, O, CB)
        Args:
            R:    (N, L, 3, 3).
            p:    (N, L, 3).
            s:    (N, L).
            res_feat:   (N, L, res_dim).
            pair_feat:  (N, L, L, pair_dim).
            mask_generate:    (N, L).
            mask_res:       (N, L).
        Returns:
            pred_p:     (N,L,3)
            pred_s:     (N,L,20)
        """
        N, L = mask_res.size()
        t_res_embed = self.t_res_embedding(t.unsqueeze(1))
        t_pair_embed = self.t_pair_embedding(t.unsqueeze(1))

        res_feat = self.res_feat_mixer(torch.cat([res_feat, self.current_sequence_embedding(s)], dim=-1)) # [Important] Incorporate sequence at the current step.
        res_feat = res_feat + t_res_embed[:, None, :]
        pair_feat = pair_feat + t_pair_embed[:, None, None,  :]
        # in_feat = self.encoder(R, p, res_feat, pair_feat, mask_res)
        in_feat = self.reipa(R, p, res_feat, pair_feat, mask_res,mask_generate)  # (N, L, res_feat_dim)    #use  this
        in_feat=self.abtransformer(mask_res=mask_res,fragment_type=fragment_type,res_feat=in_feat)

        vel_s = self.seq_net(in_feat)  # (N, L, 20)
        vel_s = torch.where(mask_generate[..., None].expand_as(vel_s), vel_s, torch.zeros_like(vel_s))

        vel_crd = self.crd_net(in_feat)
        vel_pos = apply_rotation_to_vector(R, vel_crd)
        vel_pos = torch.where(mask_generate[:, :, None].expand_as(vel_pos), vel_pos, torch.zeros_like(vel_pos))

        vel_qua = self.quaternion_net(in_feat)
        vel_qua = torch.where(mask_generate[:, :, None].expand_as(vel_qua), vel_qua, torch.zeros_like(vel_qua))
        vel_qua[:, :, 0] = 1

        return vel_s, vel_pos, vel_qua


class RectFlowGenerator(nn.Module):

    def __init__(
        self, 
        res_feat_dim, 
        pair_feat_dim, 
        num_steps, 
        eps_net_opt={}, 
        trans_rot_opt={}, 
        trans_pos_opt={}, 
        trans_seq_opt={},
        position_mean=[0.0, 0.0, 0.0],
        position_scale=[10.0],
    ):
        super().__init__()
        self.pred_net = PredictNet(res_feat_dim, pair_feat_dim, **eps_net_opt)
        self.num_steps = 100
        #self.num_steps = num_steps
        self.trans_seq = AminoacidCategoricalTransition(num_steps)
        self.trans_pos = PositionTransition(num_steps)
        self.trans_qua = QuaternionTransition(num_steps)

        self.register_buffer('position_mean', torch.FloatTensor(position_mean).view(1, 1,  -1))
        self.register_buffer('position_scale', torch.FloatTensor(position_scale).view(1, 1,  -1))
        self.register_buffer('_dummy', torch.empty([0, ]))

    def _normalize_position(self, p):
        p_norm = (p - self.position_mean) / self.position_scale
        return p_norm

    def _unnormalize_position(self, p_norm):
        p = p_norm * self.position_scale + self.position_mean
        return p

    def forward(self, R_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res, mask_anchor, R_template, p_template, s_template, mask_template_generate,\
        denoise_structure, denoise_sequence, template_enable=False, t=None, pdbid=None, fragment_type=None):
        N, L = res_feat.shape[:2]
        t = torch.rand((N,), device=self._dummy.device) 
        p_0 = self._normalize_position(p_0)
        q_0 = rotation_to_quaternion(R_0)

        p_template = torch.where(template_enable[..., None, None], self._normalize_position(p_template), torch.zeros_like(p_template))
        q_template = rotation_to_quaternion(R_template)
        q_interp, q_init = self.trans_qua.interpolate(q_0, mask_generate, t, mask_template_generate=mask_template_generate, template_enable=template_enable, q_template=q_template, pdbid=pdbid)

        R_interp = quaternion_to_rotation_matrix(q_interp)
        R_interp = torch.where(mask_generate[..., None, None].expand_as(R_0), R_interp, R_0)
        
        if torch.isnan(torch.masked_select(R_interp, mask_generate[..., None, None])).any():
            logging.warning(f'none detected.')
            R_interp = torch.where(torch.isnan(R_interp), torch.zeros_like(R_interp), R_interp)
        

        p_interp, p_init = self.trans_pos.interpolate(p_0, mask_generate, t, mask_template_generate=mask_template_generate, template_enable=template_enable, p_template=p_template)

        s_0 = clampped_one_hot(s_0, num_classes=20).float()
        s_template = clampped_one_hot(s_template, num_classes=20).float()
        
        s_interp, s_init = self.trans_seq.interpolate(s_0, mask_generate, t, mask_template_generate=mask_template_generate, x_template=s_template, template_enable=template_enable)
        vel_s, vel_pos, vel_qua = self.pred_net(
            R_interp, p_interp, s_interp, t, res_feat, pair_feat, mask_generate, mask_res, fragment_type=fragment_type
            )   
        
        loss_dict = {}
      
        loss_seq = F.mse_loss(s_0 - s_init, vel_s, reduction='none').mean(dim=-1)
        loss_seq = (loss_seq * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['seq'] = loss_seq

        loss_pos = F.mse_loss(p_0-p_init, vel_pos, reduction='none').mean(dim=-1)
        loss_pos = (loss_pos * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['pos'] = loss_pos

        vel_u = quaternion_to_rotation_matrix(vel_qua)
        R_next = R_interp @ vel_u
        R_next = torch.where(mask_generate[..., None, None].expand_as(R_next), R_next, R_0)
        
        loss_qua = rotation_matrix_cosine_loss(R_next, R_0)
        loss_qua = (loss_qua * mask_generate).sum() / (mask_generate.sum().float()+1e-8)
        loss_dict['qua'] = loss_qua
        
        return loss_dict

    def lambda_schedule(self, t,scope=0.2):
        # t 从 60 到 80
        if t < 65 or t > 85:
            return 0.0
        # 比如在 60 步从 0 开始，到 70 步增加到 0.2，最后再减小
        # 这样可以避免结构突变
        progress = (t - 65) / (85 - 65)
        return scope * torch.sin(torch.tensor(progress * 3.1415)) # 弧形增减

    @torch.no_grad()
    def sample(
        self, 
        R, p, s, 
        res_feat, pair_feat, 
        mask_generate, mask_res, mask_anchor, 
        sample_structure=True, sample_sequence=True,
        pbar=False, template_enable=True, R_template=None,
        p_template=None, s_template=None, mask_template_generate=None, fragment_type=None, batch_ref=None, batch_id=None, data_cropped=None, data_variant=None, log_dir=None, tag=None,
        single=False,multi=True,scope=0.2, save_traj_pdb=False, write_energy_csv=True,
    ):
        N, L = p.shape[:2]
        p = self._normalize_position(p)
        
        if template_enable:
            p_template = self._normalize_position(p_template)

        template_enable = False
        
        if template_enable:
            R_template = torch.masked_select(R_template, mask_template_generate[..., None, None])
            R_init = R.clone().masked_scatter(mask_generate[..., None, None].expand_as(R), R_template)
        else:
            v_init = random_uniform_so3([N, L], device=R.device)
            R_init = so3vec_to_rotation(v_init)
            R_init = torch.where(mask_generate[..., None, None].expand_as(R), R_init, R)
        
        if template_enable:
            p_template = torch.masked_select(p_template, mask_template_generate[..., None])
            p_init = p.clone().masked_scatter(mask_generate[..., None].expand_as(p), p_template)
            e_rand = torch.randn_like(p_init)
            p_init = e_rand + p_init
            p_init = torch.where(mask_generate[..., None].expand_as(p), p_init, p)
        else:
            aa_mask = p.norm(dim=-1)
            aa_mask = ~(aa_mask==0)
            context_mask = torch.logical_and(aa_mask, ~mask_generate)
            p_avg = (p*context_mask[:, :, None]).sum(dim=1) / context_mask.sum(dim=1)[:, None]
            e_rand = torch.randn_like(p)
            p_init = e_rand + p_avg.detach().clone()[:, None, :]
            p_init = torch.where(mask_generate[..., None].expand_as(p), p_init, p)

        template_enable = True

        s = clampped_one_hot(s, num_classes=20).float()
        if template_enable:
            s_template = clampped_one_hot(s_template, num_classes=20).float()
        if template_enable:
            s_template = torch.masked_select(s_template, mask_template_generate[..., None])
            s_init = s.clone().masked_scatter(mask_generate[..., None], s_template)
            s_init = s_init + torch.randn_like(s_init, device=s_init.device)
            s_init = torch.where(mask_generate[..., None], s_init, s)
        else:
            s_init = torch.randn_like(s, device=s.device)
            s_init = torch.where(mask_generate[..., None], s_init, s)

        return_c_init = torch.argmax(s_init, dim=-1)
        return_v_init = rotation_to_so3vec(R_init.detach().clone())
        return_p_init = self._unnormalize_position(p_init)

        traj = {0: (return_v_init, return_p_init, return_c_init, s_init, p_init, R_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        else:  
            pbar = lambda x: x 

        dt = 1./self.num_steps

        for t in pbar(range(0, self.num_steps)):
            
            return_v_next, return_p_next, return_c_next, s_t, p_t, R_t = traj[t]
            #
            B = s_t.size(0)
            guidance_active = 65 <= t <= 85
            guidance_structures = [] if guidance_active else None
            traj_save_dir = os.path.join(log_dir, tag, "traj_pdb") if (guidance_active or save_traj_pdb) else None
            if traj_save_dir is not None:
                os.makedirs(traj_save_dir, exist_ok=True)
            if guidance_active:
                for i in range(B):
                    aa_single = return_c_next[i:i + 1]

                    pos_atom_new, mask_atom_new = reconstruct_backbone_partially(
                        pos_ctx=batch_ref['pos_heavyatom'][i:i + 1],
                        R_new=so3vec_to_rotation(return_v_next[i:i + 1]),
                        t_new=return_p_next[i:i + 1],
                        aa=aa_single,
                        chain_nb=batch_ref['chain_nb'][i:i + 1],
                        res_nb=batch_ref['res_nb'][i:i + 1],
                        mask_atoms=batch_ref['mask_heavyatom'][i:i + 1],
                        mask_recons=batch_ref['generate_flag'][i:i + 1],
                    )

                    pos_atom_new = pos_atom_new.cpu()
                    mask_atom_new = mask_atom_new.cpu()
                    aa_single = aa_single.cpu()

                    data_tmpl = data_variant
                    aa_full = apply_patch_to_tensor(data_tmpl['aa'], aa_single[0], data_cropped['patch_idx'])
                    mask_ha = apply_patch_to_tensor(data_tmpl['mask_heavyatom'], mask_atom_new[0], data_cropped['patch_idx'])
                    pos_ha = apply_patch_to_tensor(
                        data_tmpl['pos_heavyatom'],
                        pos_atom_new[0] + batch_ref['origin'][i].view(1, 1, 3).cpu(),
                        data_cropped['patch_idx']
                    )

                    pdb_data = {
                        'chain_nb': data_tmpl['chain_nb'],
                        'chain_id': data_tmpl['chain_id'],
                        'resseq': data_tmpl['resseq'],
                        'icode': data_tmpl['icode'],
                        'aa': aa_full,
                        'mask_heavyatom': mask_ha,
                        'pos_heavyatom': pos_ha,
                    }
                    save_path = None
                    if save_traj_pdb:
                        save_path = os.path.join(
                            traj_save_dir,
                            f"batch{batch_id:02d}_sample{i:02d}_step{t:03d}.pdb"
                        )
                    guidance_structures.append(save_pdb(pdb_data, path=save_path))
            #
            t_tensor = torch.ones((N,), device=self._dummy.device) * t / self.num_steps + 1e-3
            vel_s, vel_pos, vel_qua = self.pred_net(
                R_t, p_t, s_t, t_tensor, res_feat, pair_feat, mask_generate, mask_res, fragment_type=fragment_type
            )

            s_next = s_t + dt * vel_s
            # p_next = p_t + dt * vel_pos
            ##Energy 

            if guidance_active:
                # 1. 获取物理梯度
                # 注意：物理上应该减去梯度（向能量低的方向走），所以后面用减号
                grads_batch, grads_list, meta_list, h3_ranges = run_energy_guidance(
                    batch_id,
                    traj_save_dir,
                    t,
                    batch_size=p_t.shape[0],
                    device="cuda",
                    write_csv=write_energy_csv,
                    structure_sources=guidance_structures,
                )
                
                # 2. 计算模型预测的位移项：dt * vel_pos
                drift_step = dt * vel_pos
                p_next = p_t + drift_step

                # 3. 计算引导项的量级 (Scaling)
                # 计算模型位移的平均模长，作为基准
                drift_norm = torch.mean(torch.norm(drift_step, dim=-1, keepdim=True))
                
                # 设置引导系数 lambda (建议从 0.1 到 0.5 尝试)
                # lam = self.lambda_schedule(t) # 或者直接给个常数如 0.2
                lam=0.2
                lam=self.lambda_schedule(t,scope).to(p_t.device)

                 # 5. 处理梯度方向与范围
                if single:
                    # 【Single 模式】：只保留 H3 区域的梯度，其余清零
                    # h3_only_grads = torch.zeros_like(grads_batch)
                    h3_only_grads =[]
                    for i in range(grads_batch.shape[0]):
                        start, end = h3_ranges[i]
                        # 只拷贝 H3 范围内的梯度
                        h3_only_grads.append(grads_batch[i, start:end])
                    target_grads = torch.cat(h3_only_grads, dim=0).reshape(-1, 3)
                    mode_name = "CDR-H3 Only"
                else:
                    # 【Multi 模式】：使用所有 CDR (H1,H2,H3,L1,L2,L3) 的梯度
                    target_grads = grads_batch.reshape(-1, 3)
                    mode_name = "All CDRs"

                # 6. 归一化并对齐量级
                # a) 计算梯度模长
                gnorm = torch.norm(target_grads, dim=-1, keepdim=True) + 1e-8
                
                # b) 归一化方向并乘以目标强度
                # 注意：物理上我们要减去梯度 (minus gradient) 才能降低能量
                guidance_step = - (target_grads / gnorm) * drift_norm * lam
                
                # 7. 应用更新到 p_next
                # p_next[mask_generate] 对应的点集必须与 grads_batch 的残基顺序严格一致
                p_next[mask_generate] = p_next[mask_generate] + guidance_step

                print(f"Step {t}: [{mode_name}] Energy Guidance applied. DriftNorm: {drift_norm.item():.4f}, Lam: {lam.item():.4f}")

            else:
                # --- 非引导阶段，使用标准更新 ---
                p_next = p_t + dt * vel_pos
            ##
            vel_u = quaternion_to_rotation_matrix(vel_qua)
            R_next = R_t@vel_u
            v_next = rotation_to_so3vec(R_next)

            return_c_next = torch.argmax(s_next, dim=-1)
            return_v_next = v_next.detach().clone()
            return_p_next = self._unnormalize_position(p_next).detach().clone()

            traj[t+1] = (return_v_next, return_p_next, return_c_next, s_next.detach().clone(), p_next.detach().clone(), R_next.detach().clone())
            traj[t] = tuple(x.cpu() for x in traj[t])
        
        reverse_traj = {}
        for t in range(0, self.num_steps+1):
            reverse_traj[self.num_steps - t] = traj[t]

        return reverse_traj
