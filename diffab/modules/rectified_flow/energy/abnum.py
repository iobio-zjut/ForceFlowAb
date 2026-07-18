import contextlib
import io
import os
import csv
import argparse
import time
import multiprocessing as mp  # 410

from Bio import PDB
from Bio.PDB import Selection, PDBParser
from Bio.Data import SCOPData
import abnumber
# from diffab.modules.diffusion.energy_guidance import EnergyGuidance
from diffab.modules.rectified_flow.energy.energy_guidance import EnergyGuidance
# from energy_guidance import EnergyGuidance
import torch


PDB_PARSER = PDBParser(QUIET=True)
CDR_H = [(26, 32), (52, 56), (95, 102)]
CDR_L = [(24, 34), (50, 56), (89, 97)]
CHAIN_TYPE_ORDER = ("H", "L", "Ag")
RING_RESIDUES = {"PHE", "TYR"}
R_C_ATOMS = {"CD1", "CE2"}
CDR_ALIASES = {
    "H1": "H_CDR1",
    "H_CDR1": "H_CDR1",
    "HCDR1": "H_CDR1",
    "CDRH1": "H_CDR1",
    "H2": "H_CDR2",
    "H_CDR2": "H_CDR2",
    "HCDR2": "H_CDR2",
    "CDRH2": "H_CDR2",
    "H3": "H_CDR3",
    "H_CDR3": "H_CDR3",
    "HCDR3": "H_CDR3",
    "CDRH3": "H_CDR3",
    "L1": "L_CDR1",
    "L_CDR1": "L_CDR1",
    "LCDR1": "L_CDR1",
    "CDRL1": "L_CDR1",
    "L2": "L_CDR2",
    "L_CDR2": "L_CDR2",
    "LCDR2": "L_CDR2",
    "CDRL2": "L_CDR2",
    "L3": "L_CDR3",
    "L_CDR3": "L_CDR3",
    "LCDR3": "L_CDR3",
    "CDRL3": "L_CDR3",
}


def _normalize_chain_id(chain_id):
    if chain_id is None:
        return None
    chain_id = str(chain_id).strip()
    if chain_id == "" or chain_id.lower() in {"none", "null", "-"}:
        return None
    return chain_id


def _normalize_selected_chains(selected_chains):
    if not selected_chains:
        return None
    if isinstance(selected_chains, dict):
        heavy = selected_chains.get("heavy")
        light = selected_chains.get("light")
    elif isinstance(selected_chains, (tuple, list)):
        heavy = selected_chains[0] if len(selected_chains) > 0 else None
        light = selected_chains[1] if len(selected_chains) > 1 else None
    else:
        heavy, light = selected_chains, None

    heavy = _normalize_chain_id(heavy)
    light = _normalize_chain_id(light)
    if heavy is None and light is None:
        return None
    return {"heavy": heavy, "light": light}


def _apply_selected_chains(chain_map, selected_chains):
    selected_chains = _normalize_selected_chains(selected_chains)
    if selected_chains is None:
        return chain_map

    heavy = selected_chains["heavy"]
    light = selected_chains["light"]
    filtered = {}
    for cid in chain_map:
        if heavy is not None and cid == heavy:
            filtered[cid] = "H"
        elif light is not None and cid == light:
            filtered[cid] = "L"
        else:
            filtered[cid] = "Ag"
    return filtered


def _normalize_selected_cdrs(selected_cdrs):
    if selected_cdrs is None:
        return None
    if isinstance(selected_cdrs, str):
        raw_cdrs = selected_cdrs.replace(",", " ").split()
    else:
        raw_cdrs = selected_cdrs

    normalized = set()
    for cdr in raw_cdrs:
        key = str(cdr).strip().upper().replace("-", "_")
        if not key:
            continue
        if key in CDR_ALIASES:
            normalized.add(CDR_ALIASES[key])
        else:
            raise ValueError(f"Unsupported CDR name for energy guidance: {cdr!r}")
    return normalized if normalized else None


def _cdr_name(chain_type, cdr_idx):
    return f"{chain_type}_CDR{cdr_idx + 1}"


def _residue_cdr_index(residue, chain_type, selected_cdrs=None):
    if chain_type == "H":
        cdrs = CDR_H
    elif chain_type == "L":
        cdrs = CDR_L
    else:
        return None

    num, ins = residue.id[1], residue.id[2].strip()
    for cdr_idx, (start, end) in enumerate(cdrs):
        in_range = (start <= num <= end) or (num == end and ins) or (num == start and ins)
        if in_range and (selected_cdrs is None or _cdr_name(chain_type, cdr_idx) in selected_cdrs):
            return cdr_idx
    return None

def get_sequence(chain):
    residues = Selection.unfold_entities(chain, 'R')
    seq = ''.join([SCOPData.protein_letters_3to1.get(r.resname, 'X') for r in residues])
    return seq

def _get_model(structure_source):
    if isinstance(structure_source, str):
        return PDB_PARSER.get_structure('complex', structure_source)[0]

    if hasattr(structure_source, 'get_level'):
        level = structure_source.get_level()
        if level == 'S':
            return structure_source[0]
        if level == 'M':
            return structure_source

    raise TypeError(f"Unsupported structure source type: {type(structure_source)!r}")

def classify_chains(structure_source, verbose=False):
    model = _get_model(structure_source)

    chain_types = {}
    stdout_sink = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    with stdout_sink:
        for chain in model:
            seq = get_sequence(chain)
            try:
                ab = abnumber.Chain(seq, scheme='chothia')
                ctype = ab.chain_type  # 'H', 'K', or 'L'
                if ctype == 'K':
                    ctype = 'L'
                chain_types[chain.id] = ctype
                if verbose:
                    print(f"[INFO] Chain {chain.id} classified as {ctype}")
            except abnumber.ChainParseError:
                chain_types[chain.id] = 'Ag'
                if verbose:
                    print(f"[INFO] Chain {chain.id} classified as Antigen (Ag)")
    return chain_types

def reorder_and_renumber_pdb(structure_source, chain_map):
    model = _get_model(structure_source)

    chain_map = {cid: ('L' if t == 'K' else t) for cid, t in chain_map.items()}
    order = [cid for t in CHAIN_TYPE_ORDER for cid, tp in chain_map.items() if tp == t]

    reordered_model = PDB.Model.Model(0)
    renum_model = PDB.Model.Model(0)
    new_chain_map = {}

    for new_id, old_id in zip(['A', 'B', 'C'], order):
        old_chain = model[old_id]
        ctype = chain_map[old_id]
        reordered_chain = PDB.Chain.Chain(new_id)
        renum_chain = PDB.Chain.Chain(new_id)
        residues = [res for res in old_chain.get_residues() if not res.id[0].strip()]

        if ctype == 'Ag' and residues:
            first_resnum = residues[0].id[1]
            offset = first_resnum - 1 if first_resnum != 1 else 0
        else:
            offset = 0

        res_id = 1
        for res in residues:
            het, resseq, icode = res.id
            reordered_res = res.copy()
            if offset != 0:
                reordered_res.id = (het, resseq - offset, icode)
            reordered_chain.add(reordered_res)

            renum_res = reordered_res.copy()
            renum_res.id = (' ', res_id, ' ')
            renum_chain.add(renum_res)
            res_id += 1

        reordered_model.add(reordered_chain)
        renum_model.add(renum_chain)
        new_chain_map[new_id] = ctype

    return new_chain_map, reordered_model, renum_model


def generate_cdr_mask(model, chain_map, selected_cdrs=None):#三个维度顺序从H L AG
    selected_cdrs = _normalize_selected_cdrs(selected_cdrs)
    normalized_chain_map = {cid: ('L' if tp == 'K' else tp) for cid, tp in chain_map.items()}
    order = [cid for t in CHAIN_TYPE_ORDER for cid, tp in normalized_chain_map.items() if tp == t]

    masks, lengths = [], []
    cdr_h_lengths = [0, 0, 0]
    cdr_l_lengths = [0, 0, 0]
    for cid in order:
        chain = model[cid]
        tp = normalized_chain_map[cid]
        res = list(chain.get_residues())
        mask = torch.zeros(len(res), 1, dtype=torch.float32)
        if tp in ('H', 'L'):
            for i, r in enumerate(res):
                cdr_idx = _residue_cdr_index(r, tp, selected_cdrs)
                if cdr_idx is not None:
                    mask[i] = 1
                    if tp == 'H':
                        cdr_h_lengths[cdr_idx] += 1
                    else:
                        cdr_l_lengths[cdr_idx] += 1
        masks.append(mask)
        lengths.append(len(res))

    R = max(lengths)
    masks = [torch.cat([m, torch.zeros(R - len(m), 1, dtype=m.dtype)], 0) for m in masks]
    mask_tensor = torch.stack(masks).unsqueeze(0)

    mask_tensor = torch.cat([torch.zeros_like(mask_tensor[:, :, :1, :]), mask_tensor], dim=2)

    return mask_tensor, masks[:len(order)], model, CDR_H, CDR_L, cdr_h_lengths, cdr_l_lengths

def generate_mask_atom(chain_map,atnames,model,CDR_H,CDR_L,selected_cdrs=None):#按pdb顺序来的
    selected_cdrs = _normalize_selected_cdrs(selected_cdrs)
    if isinstance(atnames[0], list):
        atnames = atnames[0]
    N = len(atnames)
    mask_atom = torch.zeros(N, 1, dtype=torch.float32)
    atom_idx = 0
    for chain in model:
        ctype = chain_map.get(chain.id, 'Ag')
        for res in chain.get_residues():
            if res.id[0].strip():
                continue
            in_cdr = _residue_cdr_index(res, ctype, selected_cdrs) is not None
            for atom in res.get_atoms():
                if atom_idx >= N:
                    break
                if in_cdr and atom.get_name() == 'CA':
                    mask_atom[atom_idx] = 1
                atom_idx += 1
        if atom_idx >= N:
            break

    return mask_atom.unsqueeze(0)

def build_madrax_inputs(chain_map, reordered_model, renum_model, selected_cdrs=None):
    selected_cdrs = _normalize_selected_cdrs(selected_cdrs)
    coords = []
    atnames = []
    mask_atom = []

    for chain_id in sorted(new_chain.id for new_chain in renum_model):
        reordered_chain = reordered_model[chain_id]
        renum_chain = renum_model[chain_id]
        ctype = chain_map.get(chain_id, 'Ag')

        reordered_residues = list(reordered_chain.get_residues())
        renum_residues = list(renum_chain.get_residues())
        for reordered_res, renum_res in zip(reordered_residues, renum_residues):
            renum_rid = renum_res.id[1]
            resname = renum_res.resname.strip()
            in_cdr = _residue_cdr_index(reordered_res, ctype, selected_cdrs) is not None
            rc_coords = {}

            for atom in renum_res.get_atoms():
                atom_name = atom.get_name().strip()
                if atom_name.startswith('H'):
                    continue

                coords.append(atom.coord.tolist())
                atnames.append(f"{resname}_{renum_rid}_{atom_name}_{chain_id}_0_0")
                mask_atom.append(1.0 if in_cdr and atom_name == 'CA' else 0.0)

                if resname in RING_RESIDUES and atom_name in R_C_ATOMS:
                    rc_coords[atom_name] = atom.coord

            if resname in RING_RESIDUES and len(rc_coords) == 2:
                rc_coord = (rc_coords["CD1"] + rc_coords["CE2"]) / 2.0
                coords.append(rc_coord.tolist())
                atnames.append(f"{resname}_{renum_rid}_RC_{chain_id}_0_0")
                mask_atom.append(0.0)

    coords_tensor = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)
    mask_atom_tensor = torch.tensor(mask_atom, dtype=torch.float32).view(1, -1, 1)
    return coords_tensor, [atnames], mask_atom_tensor



def _preprocess_single_for_energy(args):  # 410
    if len(args) == 5:
        i, batch_id, save_path, t, structure_source = args  # 410
        selected_chains = None
        selected_cdrs = None
    else:
        i, batch_id, save_path, t, structure_source, selected_chains, selected_cdrs = args  # 410
    pdb_dir = os.path.join(save_path, f"batch{batch_id:02d}_sample{i:02d}_step{t:03d}.pdb")  # 410
    source = structure_source if structure_source is not None else pdb_dir  # 410

    chain_map = classify_chains(source)  # 410
    chain_map = _apply_selected_chains(chain_map, selected_chains)  # 410
    chain_map, reordered_model, renum_model = reorder_and_renumber_pdb(source, chain_map)  # 410
    cdr_mask, _, _, _, _, cdr_h_lengths, _ = generate_cdr_mask(reordered_model, chain_map, selected_cdrs)  # 410
    coords, atnames, mask_atom = build_madrax_inputs(chain_map, reordered_model, renum_model, selected_cdrs)  # 410

    h1_len, h2_len, h3_len = cdr_h_lengths  # 410
    h3_start = h1_len + h2_len  # 410
    cdr_ca_count = int(mask_atom[..., 0].sum().item())  # 410

    return {  # 410
        "index": i,
        "pdb_name": os.path.basename(pdb_dir),
        "coords": coords,
        "atnames": atnames,
        "mask_atom": mask_atom,
        "cdr_mask": cdr_mask,
        "h3_range": (h3_start, h3_start + h3_len),
        "cdr_ca_count": cdr_ca_count,
    }


def _choose_mp_context():  # 410
    try:
        return mp.get_context("fork")  # 410
    except ValueError:
        return mp.get_context("spawn")  # 410

def run_energy_guidance(
    batch_id,
    save_path,
    t,
    batch_size=16,
    device="cuda",
    write_csv=True,
    return_details=False,
    structure_sources=None,
    preprocess_workers=None,  # 410
    selected_chains=None,  # 410
    selected_cdrs=None,  # 410
):
    run_t0 = time.perf_counter()  # 410
    force_accum = []  # 410
    torque_accum = []  # 410
    grads_list = [] if return_details else None
    meta_list = [] if (return_details or write_csv) else None
    h3_ranges = []
    eg = EnergyGuidance(device=device)

    worker_inputs = []  # 410
    for i in range(batch_size):  # 410
        structure_source = structure_sources[i] if structure_sources is not None else None  # 410
        worker_inputs.append((i, batch_id, save_path, t, structure_source, selected_chains, selected_cdrs))  # 410

    if preprocess_workers is None:  # 410
        cpu_n = os.cpu_count() or 1  # 410
        preprocess_workers = min(8, max(1, cpu_n // 2))  # 410

    source_is_path_like = (structure_sources is None) or all(isinstance(x, str) for x in structure_sources)  # 410
    use_mp = preprocess_workers > 1 and source_is_path_like  # 410

    preprocess_t0 = time.perf_counter()  # 410
    if use_mp:  # 410
        ctx = _choose_mp_context()  # 410
        with ctx.Pool(processes=preprocess_workers) as pool:  # 410
            preprocessed = pool.map(_preprocess_single_for_energy, worker_inputs)  # 410
    else:
        preprocessed = [_preprocess_single_for_energy(x) for x in worker_inputs]  # 410

    preprocessed.sort(key=lambda x: x["index"])  # 410
    preprocess_sec = time.perf_counter() - preprocess_t0  # 410

    gpu_t0 = time.perf_counter()  # 410
    for item in preprocessed:  # 410
        guidance, meta = eg.compute_grad_batch(
            pdb_name=item["pdb_name"],
            cdr_mask=item["cdr_mask"],
            coords=item["coords"],
            atnames=item["atnames"],
            mask_atom=item["mask_atom"],
            return_meta=(meta_list is not None),
        )
        force_ca = guidance["force"]  # 410
        torque_ca = guidance["torque"]  # 410
        force_accum.append(force_ca)  # 410
        torque_accum.append(torque_ca)  # 410
        if grads_list is not None:
            grads_list.append({"force": force_ca, "torque": torque_ca})  # 410
        if meta_list is not None:
            meta_list.append(meta)

        h3_ranges.append(item["h3_range"])  # 410

    gpu_sec = time.perf_counter() - gpu_t0  # 410

    force_batch = torch.cat(force_accum, dim=0)  # 410
    torque_batch = torch.cat(torque_accum, dim=0)  # 410
    grads_batch = {"force": force_batch, "torque": torque_batch}  # 410
    csv_sec = 0.0  # 410
    if write_csv and meta_list is not None:
        csv_t0 = time.perf_counter()  # 410
        csv_name = f"{batch_id}_{t}_energy_gradients.csv"
        csv_path = os.path.join(save_path, csv_name)
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pdb_name", "Energy", "NonZeroCount", "Gradients"])

            for i, meta in enumerate(meta_list):
                grad = force_accum[i][0]  # 410
                nz_idx = (grad.abs().sum(dim=-1) > 0).nonzero(as_tuple=False).flatten()
                gradients_str = "; ".join(
                    f"{v[0].item():.6f} {v[1].item():.6f} {v[2].item():.6f}"
                    for v in grad[nz_idx]
                )
                writer.writerow([meta["pdb_name"], meta["E_total"], int(nz_idx.numel()), gradients_str])
        csv_sec = time.perf_counter() - csv_t0  # 410

    total_sec = time.perf_counter() - run_t0  # 410
    per_sample = total_sec / max(batch_size, 1)  # 410
    print(  # 410
        f"[EnergyGuidance][Timing] batch={batch_id} step={t} samples={batch_size} "  # 410
        f"preprocess={preprocess_sec:.3f}s gpu={gpu_sec:.3f}s csv={csv_sec:.3f}s "  # 410
        f"total={total_sec:.3f}s per_sample={per_sample:.3f}s mp={use_mp} workers={preprocess_workers}"  # 410
    )

    return grads_batch, (grads_list or []), (meta_list or []), h3_ranges




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug EnergyGuidance independently.")
    parser.add_argument("--batch_id", type=int, default=0,
                        help="batch_id")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Directory containing sampleXX_stepXXX.pdb files")
    parser.add_argument("--t", type=int, default=100,
                        help="Diffusion timestep t (used in filename)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Number of samples to load (default 16)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on: 'cuda' or 'cpu'")
    parser.add_argument("--preprocess_workers", type=int, default=None,  # 410
                        help="CPU workers for preprocessing; <=1 means serial.")  # 410
    args = parser.parse_args()
    #######
    run_energy_guidance(batch_id =args.batch_id,save_path = args.save_path,
    t =args.t,
    batch_size = args.batch_size,
    device = args.device,
    preprocess_workers = args.preprocess_workers,  # 410
    )


    

