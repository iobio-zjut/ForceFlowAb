import torch
import torch.nn as nn
from esm import pretrained, Alphabet
from esm.pretrained import load_model_and_alphabet_local
import torch.nn.functional as F

from diffab.modules.common.geometry import construct_3d_basis, global_to_local, get_backbone_dihedral_angles
from diffab.modules.common.layers import AngularEncoding
from diffab.utils.protein.constants import BBHeavyAtom, AA


model, alphabet = load_model_and_alphabet_local("/share/home/zhanglab/lzh/esm/ESM_model/esm2_t33_650M_UR50D.pt")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
standard_toks = [alphabet.get_idx(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"]
your_to_esm = torch.tensor(standard_toks + [alphabet.get_idx('<unk>')], dtype=torch.long).to(device)


def get_esm2_representations(batch_aa: torch.Tensor, 
                            return_mean=True, 
                            return_per_residue=False):
    """
    Args:
        batch_aa: (B, L) LongTensor，取值 0~20（你的词汇表）
        return_mean: 是否返回整蛋白均值池化表征
        return_per_residue: 是否返回每残基表征
    Returns:
        根据需求返回 mean_emb (B, D) 和/或 per_res_emb (B, L, D)
    """
    B, L = batch_aa.shape
    batch_aa = batch_aa.to(device)
    
    # Step1: 映射到 ESM 的真实 token id
    esm_tokens = your_to_esm[batch_aa]  # (B, L)，现在取值是 3 或 4~23
    
    # Step2: 加上 ESM 必须的 <cls> 和 <eos>
    cls_tok = alphabet.cls_idx  # 0
    eos_tok = alphabet.eos_idx  # 2
    prefix = torch.full((B, 1), cls_tok, dtype=torch.long, device=device)
    suffix = torch.full((B, 1), eos_tok, dtype=torch.long, device=device)
    
    esm_input = torch.cat([prefix, esm_tokens, suffix], dim=1)  # (B, L+2)
    
    # Step3: 前向（只取最后一层）
    with torch.no_grad():
        results = model(esm_input, repr_layers=[33])  # 33 是最后一层（对于 t33 模型）
        embeddings = results["representations"][33]  # (B, L+2, D)
    
    # Step4: 去掉 <cls> 和 <eos>，得到每残基表征
    per_residue = embeddings[:, 1:-1, :]  # (B, L, D)
    
    outputs = {}
    if return_per_residue:
        outputs["per_residue"] = per_residue
    
    if return_mean:
        # mean pooling（推荐，比只取 <cls> 更稳定）
        outputs["mean"] = per_residue.mean(dim=1)  # (B, D)
    
    # 也可以返回 <cls> 表征（很多论文用这个）
    # outputs["cls"] = embeddings[:, 0, :]
    # print('YES ESM')
    return outputs


class ResidueEmbedding(nn.Module):

    def __init__(self, feat_dim, max_num_atoms, max_aa_types=22):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types
        self.aatype_embed = nn.Embedding(self.max_aa_types, feat_dim)
        self.dihed_embed = AngularEncoding()
        self.type_embed = nn.Embedding(10, feat_dim, padding_idx=0)    # 1: Heavy, 2: Light, 3: Ag
        infeat_dim = feat_dim + (self.max_aa_types*max_num_atoms*3) + self.dihed_embed.get_out_dim(3) + feat_dim+1280
        self.mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim * 2), nn.ReLU(),
            nn.Linear(feat_dim * 2, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, aa, res_nb, chain_nb, pos_atoms, mask_atoms, fragment_type, structure_mask=None, sequence_mask=None):
        """
        Args:
            aa:         (N, L).
            res_nb:     (N, L).
            chain_nb:   (N, L).
            pos_atoms:  (N, L, A, 3).
            mask_atoms: (N, L, A).
            fragment_type:  (N, L).
            structure_mask: (N, L), mask out unknown structures to generate.
            sequence_mask:  (N, L), mask out unknown amino acids to generate.
        """
        N, L = aa.size()
        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA] # (N, L)

        # Remove other atoms
        pos_atoms = pos_atoms[:, :, :self.max_num_atoms]
        mask_atoms = mask_atoms[:, :, :self.max_num_atoms]

        # Amino acid identity features
        if sequence_mask is not None:
            # Avoid data leakage at training time
            aa = torch.where(sequence_mask, aa, torch.full_like(aa, fill_value=AA.UNK))
            # print(aa.shape)
            # print(aa)
            # assert  False,'debug'
        reps = get_esm2_representations(aa, return_mean=True, return_per_residue=True)
        per_residue=reps["per_residue"] #(N, L, 1280)
        aa_feat = self.aatype_embed(aa) # (N, L, feat)

        # Coordinate features
        R = construct_3d_basis(
            pos_atoms[:, :, BBHeavyAtom.CA], 
            pos_atoms[:, :, BBHeavyAtom.C], 
            pos_atoms[:, :, BBHeavyAtom.N]
        )
        t = pos_atoms[:, :, BBHeavyAtom.CA]
        crd = global_to_local(R, t, pos_atoms)    # (N, L, A, 3)
        crd_mask = mask_atoms[:, :, :, None].expand_as(crd)
        crd = torch.where(crd_mask, crd, torch.zeros_like(crd))

        aa_expand  = aa[:, :, None, None, None].expand(N, L, self.max_aa_types, self.max_num_atoms, 3)
        rng_expand = torch.arange(0, self.max_aa_types)[None, None, :, None, None].expand(N, L, self.max_aa_types, self.max_num_atoms, 3).to(aa_expand)
        place_mask = (aa_expand == rng_expand)
        crd_expand = crd[:, :, None, :, :].expand(N, L, self.max_aa_types, self.max_num_atoms, 3)
        crd_expand = torch.where(place_mask, crd_expand, torch.zeros_like(crd_expand))
        crd_feat = crd_expand.reshape(N, L, self.max_aa_types*self.max_num_atoms*3)
        if structure_mask is not None:
            # Avoid data leakage at training time
            crd_feat = crd_feat * structure_mask[:, :, None]

        # Backbone dihedral features
        bb_dihedral, mask_bb_dihed = get_backbone_dihedral_angles(pos_atoms, chain_nb=chain_nb, res_nb=res_nb, mask=mask_residue)
        dihed_feat = self.dihed_embed(bb_dihedral[:, :, :, None]) * mask_bb_dihed[:, :, :, None]  # (N, L, 3, dihed/3)
        dihed_feat = dihed_feat.reshape(N, L, -1)
        if structure_mask is not None:
            # Avoid data leakage at training time
            dihed_mask = torch.logical_and(
                structure_mask,
                torch.logical_and(
                    torch.roll(structure_mask, shifts=+1, dims=1), 
                    torch.roll(structure_mask, shifts=-1, dims=1)
                ),
            )   # Avoid slight data leakage via dihedral angles of anchor residues
            dihed_feat = dihed_feat * dihed_mask[:, :, None]

        # Type feature
        type_feat = self.type_embed(fragment_type) # (N, L, feat)

        out_feat = self.mlp(torch.cat([aa_feat, crd_feat, dihed_feat, type_feat,per_residue], dim=-1)) # (N, L, F)
        out_feat = out_feat * mask_residue[:, :, None]
        return out_feat
