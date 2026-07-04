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
from diffab.modules.encoders.MOE import ABTransformer   #316
from diffab.modules.encoders.Triangle import TriangleUpdate   #316
from diffab.modules.encoders.single_update import SingleUpdate   #316
# from diffab.modules.encoders.transform import ABTransformer
#
import os
from diffab.modules.common.geometry import reconstruct_backbone_partially   #lll
from diffab.utils.protein.writers import save_pdb
from diffab.utils.data import *
from diffab.modules.rectified_flow.energy.abnum import run_energy_guidance
#

AA_HYDROPATHY = [
    1.8, 2.5, -3.5, -3.5, 2.8,
    -0.4, -3.2, 4.5, -3.9, 3.8,
    1.9, -3.5, -1.6, -3.5, -4.5,
    -0.8, -0.7, 4.2, -0.9, -1.3,
]
AA_HYDROPATHY = [x / 4.5 for x in AA_HYDROPATHY]

AA_CHARGE = [
    0.0, 0.0, -1.0, -1.0, 0.0,
    0.0, 0.1, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
]


def build_cdr_target_one_hot(cdr_sequence, mask_generate):
    if cdr_sequence is None:
        return None

    mask_generate = mask_generate.bool()
    num_batch, seq_len = mask_generate.shape

    if cdr_sequence.dim() == 2:
        if cdr_sequence.size(1) == seq_len:
            return clampped_one_hot(cdr_sequence, num_classes=20).float()

        cdr_one_hot = torch.zeros(
            (num_batch, seq_len, 20),
            device=cdr_sequence.device,
            dtype=torch.float32,
        )
        for batch_idx in range(num_batch):
            num_cdr = int(mask_generate[batch_idx].sum().item())
            if cdr_sequence.size(1) < num_cdr:
                raise ValueError('cdr_sequence length is shorter than the number of generated CDR residues.')
            compact_tokens = cdr_sequence[batch_idx, :num_cdr]
            cdr_one_hot[batch_idx, mask_generate[batch_idx]] = clampped_one_hot(
                compact_tokens.unsqueeze(0), num_classes=20
            ).float().squeeze(0)
        return cdr_one_hot

    if cdr_sequence.dim() == 3 and cdr_sequence.size(-1) == 20:
        if cdr_sequence.size(1) == seq_len:
            return cdr_sequence.float()

        cdr_one_hot = torch.zeros(
            (num_batch, seq_len, 20),
            device=cdr_sequence.device,
            dtype=cdr_sequence.dtype,
        )
        for batch_idx in range(num_batch):
            num_cdr = int(mask_generate[batch_idx].sum().item())
            if cdr_sequence.size(1) < num_cdr:
                raise ValueError('cdr_sequence length is shorter than the number of generated CDR residues.')
            cdr_one_hot[batch_idx, mask_generate[batch_idx]] = cdr_sequence[batch_idx, :num_cdr]
        return cdr_one_hot.float()

    raise ValueError('cdr_sequence must be shaped as (N, L), (N, num_cdr), (N, L, 20), or (N, num_cdr, 20).')


def compute_cdr_property_loss(pred_seq_prob, target_seq_one_hot, mask, property_table):
    property_table = property_table.view(1, 1, -1)
    pred_property = (pred_seq_prob * property_table).sum(dim=-1)
    target_property = (target_seq_one_hot * property_table).sum(dim=-1)
    loss = F.mse_loss(pred_property, target_property, reduction='none')
    loss = (loss * mask.float()).sum() / (mask.float().sum() + 1e-8)
    return loss


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
        # self.Triang = TriangleUpdate(dim_pairwise=pair_feat_dim)    #59
        # self.singleup=SingleUpdate(dim_single=res_feat_dim,dim_pairwise=pair_feat_dim,pair_bias_attn_heads=8,pair_bias_attn_dim_head=32)    #59
        self.reipa=REIPA(res_feat_dim, pair_feat_dim, num_layers) ##use this
        self.abtransformer = ABTransformer(
            vocab_size=res_feat_dim,
            d_model=res_feat_dim*2,
            N=6,        #3->6,  #number of layers in ABTransformer
            nheads=8,   
            dropout=0.1,
            src_input_dim=res_feat_dim,   
            tgt_input_dim=res_feat_dim,
            use_moe=True,   #317
            n_routed_experts=4,             #317
            num_experts_per_tok=2,  #317
            n_shared_experts=1, #317
            aux_loss_alpha=0.01,    #317
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

    def forward(self, R, p, s, t, res_feat, pair_feat, mask_generate, mask_res, fragment_type=None, aa_labels=None):
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
        # pair_feat=self.Triang(pair_feat)    #428    
        # res_feat=self.singleup(res_feat,pair_feat)    #428

        # in_feat = self.encoder(R, p, res_feat, pair_feat, mask_res)
        in_feat = self.reipa(R, p, res_feat, pair_feat, mask_res,mask_generate)  # (N, L, res_feat_dim)    #use  this
        # in_feat=self.abtransformer(mask_res=mask_res,fragment_type=fragment_type,res_feat=in_feat)
        mydic=self.abtransformer(
            mask_res=mask_res,
            fragment_type=fragment_type,
            res_feat=in_feat,
            aa_labels=aa_labels,
            generate_mask=mask_generate,
            return_dict=True,
        )
        aux_loss = mydic['aux_loss']
        in_feat=mydic['logits']  # (N, L, res_feat_dim)


        vel_s = self.seq_net(in_feat)  # (N, L, 20)
        vel_s = torch.where(mask_generate[..., None].expand_as(vel_s), vel_s, torch.zeros_like(vel_s))

        vel_crd = self.crd_net(in_feat)
        vel_pos = apply_rotation_to_vector(R, vel_crd)
        vel_pos = torch.where(mask_generate[:, :, None].expand_as(vel_pos), vel_pos, torch.zeros_like(vel_pos))

        vel_qua = self.quaternion_net(in_feat)
        vel_qua = torch.where(mask_generate[:, :, None].expand_as(vel_qua), vel_qua, torch.zeros_like(vel_qua))
        vel_qua[:, :, 0] = 1

        return vel_s, vel_pos, vel_qua,aux_loss


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
        self.register_buffer('aa_hydropathy', torch.tensor(AA_HYDROPATHY, dtype=torch.float32))
        self.register_buffer('aa_charge', torch.tensor(AA_CHARGE, dtype=torch.float32))
        self.register_buffer('_dummy', torch.empty([0, ]))

    def _normalize_position(self, p):
        p_norm = (p - self.position_mean) / self.position_scale
        return p_norm

    def _unnormalize_position(self, p_norm):
        p = p_norm * self.position_scale + self.position_mean
        return p

    def forward(self, R_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res, mask_anchor, R_template, p_template, s_template, mask_template_generate,\
        denoise_structure, denoise_sequence, template_enable=False, t=None, pdbid=None, fragment_type=None, cdr_sequence=None):
        N, L = res_feat.shape[:2]
        aa_labels = s_0
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
        vel_s, vel_pos, vel_qua,aux_loss = self.pred_net(
            R_interp, p_interp, s_interp, t, res_feat, pair_feat, mask_generate, mask_res,
            fragment_type=fragment_type, aa_labels=aa_labels
            )   
        
        loss_dict = {}
      
        loss_seq = F.mse_loss(s_0 - s_init, vel_s, reduction='none').mean(dim=-1)
        loss_seq = (loss_seq * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['seq'] = loss_seq

        pred_seq_prob = torch.softmax(s_init + vel_s, dim=-1)
        cdr_sequence=s_0
        cdr_target = build_cdr_target_one_hot(cdr_sequence, mask_generate)
        if cdr_target is None:
            loss_dict['hydropathy'] = loss_seq.new_zeros(())
            loss_dict['charge'] = loss_seq.new_zeros(())
        else:
            cdr_target = cdr_target.to(pred_seq_prob.device)
            loss_dict['hydropathy'] = compute_cdr_property_loss(
                pred_seq_prob,
                cdr_target,
                mask_generate,
                self.aa_hydropathy.to(pred_seq_prob.device),
            )
            loss_dict['charge'] = compute_cdr_property_loss(
                pred_seq_prob,
                cdr_target,
                mask_generate,
                self.aa_charge.to(pred_seq_prob.device),
            )

        loss_pos = F.mse_loss(p_0-p_init, vel_pos, reduction='none').mean(dim=-1)
        loss_pos = (loss_pos * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['pos'] = loss_pos

        vel_u = quaternion_to_rotation_matrix(vel_qua)
        R_next = R_interp @ vel_u
        R_next = torch.where(mask_generate[..., None, None].expand_as(R_next), R_next, R_0)
        
        loss_qua = rotation_matrix_cosine_loss(R_next, R_0)
        loss_qua = (loss_qua * mask_generate).sum() / (mask_generate.sum().float()+1e-8)
        loss_dict['qua'] = loss_qua
        loss_dict['aux'] = aux_loss
        return loss_dict

    
    def lambda_schedule(self, t, scope=0.2):
      if t < 0 or t > 99:
          return 0.0
      if t < 50:
          return torch.tensor(0.0)
      if t < 60:
          return torch.tensor(scope * (t - 50) / 10.0)
      return torch.tensor(scope)



    def _prepare_guidance_vectors(self, grads_batch, h3_ranges, single):  # 410
        if isinstance(grads_batch, dict):  # 410
            force = grads_batch.get("force", grads_batch.get("grad", None))  # 410
            torque = grads_batch.get("torque", None)  # 410
            if force is None:  # 410
                raise ValueError("grads_batch dict must contain 'force' or 'grad'.")  # 410
        elif isinstance(grads_batch, (tuple, list)) and len(grads_batch) == 2:  # 410
            force, torque = grads_batch  # 410
        else:  # 410
            force, torque = grads_batch, None  # 410

        if not torch.is_tensor(force):  # 410
            force = torch.as_tensor(force, device=self._dummy.device, dtype=torch.float32)  # 410
        if torque is None:  # 410
            torque = torch.zeros_like(force)  # 410
        elif not torch.is_tensor(torque):  # 410
            torque = torch.as_tensor(torque, device=force.device, dtype=force.dtype)  # 410

        if single:  # 410
            force_parts = []  # 410
            torque_parts = []  # 410
            for i in range(force.shape[0]):  # 410
                start, end = h3_ranges[i]  # 410
                force_parts.append(force[i, start:end])  # 410
                torque_parts.append(torque[i, start:end])  # 410
            force_flat = torch.cat(force_parts, dim=0).reshape(-1, 3)  # 410
            torque_flat = torch.cat(torque_parts, dim=0).reshape(-1, 3)  # 410
        else:  # 410
            force_flat = force.reshape(-1, 3)  # 410
            torque_flat = torque.reshape(-1, 3)  # 410

        return force_flat, torque_flat  # 410

    @torch.no_grad()
    def sample(
        self, 
        R, p, s, 
        res_feat, pair_feat, 
        mask_generate, mask_res, mask_anchor, 
        sample_structure=True, sample_sequence=True,
        pbar=False, template_enable=False, R_template=None,
        p_template=None, s_template=None, mask_template_generate=None, fragment_type=None, batch_ref=None, batch_id=None, data_cropped=None, data_variant=None, log_dir=None, tag=None,
        single=False,multi=True,scope=0.2,give_energy=True,
    ):
        self.debug=False
        N, L = p.shape[:2]
        p = self._normalize_position(p)
        
        if template_enable:
            p_template = self._normalize_position(p_template)

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
            if 89 <= t <= 99 and self.debug :
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
                    mask_ha = apply_patch_to_tensor(data_tmpl['mask_heavyatom'], mask_atom_new[0],
                                                    data_cropped['patch_idx'])
                    pos_ha = apply_patch_to_tensor(
                        data_tmpl['pos_heavyatom'],
                        pos_atom_new[0] + batch_ref['origin'][i].view(1, 1, 3).cpu(),
                        data_cropped['patch_idx']
                    )

                    traj_save_dir = os.path.join(log_dir, tag, "traj_pdb")
                    os.makedirs(traj_save_dir, exist_ok=True)
                    save_path = os.path.join(
                        traj_save_dir,
                        f"batch{batch_id:02d}_sample{i:02d}_step{t:03d}.pdb"
                    )
                    save_pdb({
                        'chain_nb': data_tmpl['chain_nb'],
                        'chain_id': data_tmpl['chain_id'],
                        'resseq': data_tmpl['resseq'],
                        'icode': data_tmpl['icode'],
                        'aa': aa_full,
                        'mask_heavyatom': mask_ha,
                        'pos_heavyatom': pos_ha,
                    }, path=save_path)
                pdb_path = traj_save_dir
            #
            t_tensor = torch.ones((N,), device=self._dummy.device) * t / self.num_steps + 1e-3
            vel_s, vel_pos, vel_qua,aux_loss = self.pred_net(
                R_t, p_t, s_t, t_tensor, res_feat, pair_feat, mask_generate, mask_res, fragment_type=fragment_type
            )

            s_next = s_t + dt * vel_s
            # p_next = p_t + dt * vel_pos
            ##Energy 

            rotation_guidance_step = None  # 410
            if 89 <= t <= 99 and self.debug :
                grads_batch, grads_list, meta_list, h3_ranges = run_energy_guidance(  # 410
                    batch_id, traj_save_dir, t, batch_size=p_t.shape[0], device="cuda"  # 410
                )
                drift_step = dt * vel_pos  # 410
                p_next = p_t + drift_step  # 410

                drift_norm = torch.mean(torch.norm(drift_step, dim=-1, keepdim=True))  # 410

                lam = self.lambda_schedule(t, scope).to(p_t.device)  # 410
                target_force, target_torque = self._prepare_guidance_vectors(grads_batch, h3_ranges, single)  # 410
                target_force = target_force.to(p_t.device)  # 410
                target_torque = target_torque.to(p_t.device)  # 410

                mode_name = "CDR-H3 Only" if single else "All CDRs"  # 410
                guidance_step = - target_force * drift_norm * lam  # 410

                tnorm = torch.norm(target_torque, dim=-1, keepdim=True) + 1e-8  # 410
                rot_drift_norm = torch.mean(torch.norm(vel_qua[..., 1:], dim=-1, keepdim=True))  # 410
                rotation_guidance_step = - (target_torque / tnorm) * rot_drift_norm * lam  # 410

            if 89 <= t <= 99 and give_energy and self.debug:
                p_next[mask_generate] = p_next[mask_generate] + guidance_step  # 410
                print(  # 410
                    f"Step {t}:give_energy={give_energy} [{mode_name}] Energy Guidance applied. "  # 410
                    f"DriftNorm: {drift_norm.item():.4f}, RotNorm: {rot_drift_norm.item():.4f}, Lam: {lam.item():.4f}"  # 410
                )
            else:
                p_next = p_t + dt * vel_pos  # 410


            ##
            vel_u = quaternion_to_rotation_matrix(vel_qua)
            R_next = R_t@vel_u
            if rotation_guidance_step is not None and rotation_guidance_step.numel() > 0:  # 410
                delta_R_guidance = so3vec_to_rotation(rotation_guidance_step)  # 410
                R_next[mask_generate] = torch.matmul(delta_R_guidance, R_next[mask_generate])  # 410
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
