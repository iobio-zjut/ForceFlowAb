from email.policy import strict
import os
import argparse
import copy
import csv
import json
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import pickle
import random

from diffab.datasets import get_dataset
from diffab.models import get_model
from diffab.modules.common.geometry import reconstruct_backbone_partially
from diffab.modules.common.so3 import so3vec_to_rotation
from diffab.utils.inference import RemoveNative
from diffab.utils.protein.writers import save_pdb
from diffab.utils.train import recursive_to
from diffab.utils.misc import *
from diffab.utils.data import *
from diffab.utils.transforms import *
from diffab.utils.inference import *

# from torch_ema import ExponentialMovingAverage
# torch.cuda.set_device(4)

MOE_OUTPUT_AA_FIELDS = [
    'variant_tag', 'variant_name', 'batch_id',
    'layer_name', 'layer_role', 'layer_idx', 'expert_idx',
    'aa_idx', 'aa_name', 'token_weight_sum', 'output_prob_sum',
    'expert_output_total', 'output_prob_frac',
]

MOE_ROUTING_FIELDS = [
    'variant_tag', 'variant_name', 'batch_id',
    'layer_name', 'layer_role', 'layer_idx', 'expert_idx',
    'cdr_token_count', 'hard_count', 'soft_count',
    'hydropathy_sum', 'charge_sum', 'hydropathy_mean', 'charge_mean',
]


def get_moe_transformer(module):
    rectflow_module = getattr(module, 'rectflow_seq_only', None)
    pred_net = getattr(rectflow_module, 'pred_net', None)
    return getattr(pred_net, 'abtransformer', None)


def reset_moe_output_aa_stats(model):
    transformer = get_moe_transformer(model)
    if transformer is not None and hasattr(transformer, 'reset_routing_output_aa_stats'):
        transformer.reset_routing_output_aa_stats()


def reset_moe_routing_stats(model):
    transformer = get_moe_transformer(model)
    if transformer is not None and hasattr(transformer, 'reset_routing_stats'):
        transformer.reset_routing_stats()


def get_energy_guidance_options(config):
    guidance_cfg = config.sampling.get('energy_guidance', {}) or {}
    if 'start_step' in guidance_cfg or 'end_step' in guidance_cfg:
        start_step = guidance_cfg.get('start_step', 90)
        end_step = guidance_cfg.get('end_step', 99)
    else:
        steps = int(guidance_cfg.get('steps', 0))
        start_step = max(0, 100 - steps)
        end_step = 99
    return {
        'energy_guidance': bool(guidance_cfg.get('enabled', False)),
        'energy_guidance_start_step': int(start_step),
        'energy_guidance_end_step': int(end_step),
        'energy_guidance_warmup_steps': int(guidance_cfg.get('warmup_steps', 0)),
    }


def append_moe_output_aa_rows(csv_path, rows):
    if len(rows) == 0:
        return
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=MOE_OUTPUT_AA_FIELDS)
        if (not file_exists) or os.path.getsize(csv_path) == 0:
            writer_csv.writeheader()
        writer_csv.writerows(rows)


def flush_moe_output_aa_csv(model, csv_path, variant, batch_id):
    transformer = get_moe_transformer(model)
    if transformer is None or not hasattr(transformer, 'pop_routing_output_aa_stats'):
        return 0
    rows = transformer.pop_routing_output_aa_stats()
    for row in rows:
        row['variant_tag'] = variant.get('tag', '')
        row['variant_name'] = variant.get('name', '')
        row['batch_id'] = batch_id
    append_moe_output_aa_rows(csv_path, rows)
    return len(rows)


def append_moe_routing_rows(csv_path, rows):
    if len(rows) == 0:
        return
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=MOE_ROUTING_FIELDS)
        if (not file_exists) or os.path.getsize(csv_path) == 0:
            writer_csv.writeheader()
        writer_csv.writerows(rows)


def flush_moe_routing_csv(model, csv_path, variant, batch_id):
    transformer = get_moe_transformer(model)
    if transformer is None or not hasattr(transformer, 'pop_routing_stats'):
        return 0
    rows = transformer.pop_routing_stats()
    for row in rows:
        row['variant_tag'] = variant.get('tag', '')
        row['variant_name'] = variant.get('name', '')
        row['batch_id'] = batch_id
    append_moe_routing_rows(csv_path, rows)
    return len(rows)


def create_data_variants(config, structure_factory, pkl_dict):
    structure = structure_factory()

    structure_id = structure['id']
    template_path = pkl_dict.get(structure_id)

    if template_path:
        # template_path = os.path.join(template_path, str(random.choice(range(100)))+'.pkl')
        template_path = template_path
    else:
        template_path = None
    if template_path and os.path.exists(template_path):
        with open (template_path, 'rb') as fin:
            template = pickle.load(fin)

    data_variants = []
    if config.mode == 'single_cdr':
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(config.sampling.cdrs)))
        for cdr_name in cdrs:
            transform = Compose([
                MaskSingleCDR(cdr_name, augmentation=False),
                MergeChains(),
            ])
            data_var = transform(structure_factory())
            template = transform(template)
            residue_first, residue_last = get_residue_first_last(data_var)
            data_variants.append({
                'data': data_var,
                'template': template,
                'name': f'{structure_id}-{cdr_name}',
                'tag': f'{cdr_name}',
                'cdr': cdr_name,
                'residue_first': residue_first,
                'residue_last': residue_last,
            })
    elif config.mode == 'multiple_cdrs':
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(config.sampling.cdrs)))
        transform = Compose([
            MaskMultipleCDRs(selection=cdrs, augmentation=False),
            MergeChains(),
        ])
        data_var = transform(structure_factory())
        template = transform(template)
        data_variants.append({
            'data': data_var,
            'template': template,
            'name': f'{structure_id}-MultipleCDRs',
            'tag': 'MultipleCDRs',
            'cdrs': cdrs,
            'residue_first': None,
            'residue_last': None,
        })
    elif config.mode == 'full':
        transform = Compose([
            MaskAntibody(),
            MergeChains(),
        ])
        data_var = transform(structure_factory())
        data_variants.append({
            'data': data_var,
            'name': f'{structure_id}-Full',
            'tag': 'Full',
            'residue_first': None,
            'residue_last': None,
        })
    elif config.mode == 'abopt':
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(config.sampling.cdrs)))
        for cdr_name in cdrs:
            transform = Compose([
                MaskSingleCDR(cdr_name, augmentation=False),
                MergeChains(),
            ])
            data_var = transform(structure_factory())
            residue_first, residue_last = get_residue_first_last(data_var)
            for opt_step in config.sampling.optimize_steps:
                data_variants.append({
                    'data': data_var,
                    'name': f'{structure_id}-{cdr_name}-O{opt_step}',
                    'tag': f'{cdr_name}-O{opt_step}',
                    'cdr': cdr_name,
                    'opt_step': opt_step,
                    'residue_first': residue_first,
                    'residue_last': residue_last,
                })
    else:
        raise ValueError(f'Unknown mode: {config.mode}.')
    return data_variants

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('index', type=int)
    parser.add_argument('-c', '--config', type=str, default='./configs/test/codesign_single.yml')
    parser.add_argument('-o', '--out_root', type=str, default='./results')
    parser.add_argument('-t', '--tag', type=str, default='')
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('--template_dict', type=str, required=True)
    args = parser.parse_args()

    with open (args.template_dict, 'rb') as fin:
        pkl_dict = pickle.load(fin)

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(args.seed if args.seed is not None else config.sampling.seed)

    # Testset
    dataset = get_dataset(config.dataset.test)
    get_structure = lambda: dataset[args.index]

    # Logging
    structure_ = get_structure()
    structure_id = structure_['id']
    tag_postfix = '_%s' % args.tag if args.tag else ''
    log_dir = get_new_log_dir(os.path.join(args.out_root, config_name + tag_postfix), prefix='%04d_%s' % (args.index, structure_['id']))
    logger = get_logger('sample', log_dir)
    logger.info('Data ID: %s' % structure_['id'])
    data_native = MergeChains()(structure_)
    save_pdb(data_native, os.path.join(log_dir, 'reference.pdb'))

    # Load checkpoint and model
    logger.info('Loading model config and checkpoints: %s' % (config.model.checkpoint))
    ckpt = torch.load(config.model.checkpoint, map_location='cpu')
    cfg_ckpt = ckpt['config']
    print(cfg_ckpt)
    model = get_model(cfg_ckpt.model).to(args.device)
    lsd = model.load_state_dict(ckpt['model'],strict=False)
    logger.info(str(lsd))

    # Make data variants
    data_variants = create_data_variants(
        config = config,
        structure_factory = get_structure,
        pkl_dict=pkl_dict
    )

    # Save metadata
    metadata = {
        'identifier': structure_id,
        'index': args.index,
        'config': args.config,
        'items': [{kk: vv for kk, vv in var.items() if kk != 'data' and kk != 'template'} for var in data_variants],
    }
    with open(os.path.join(log_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Start sampling
    collate_fn = PaddingCollate(eight=False)
    inference_tfm = [ PatchAroundAnchor(), ]
    inference_tfm = Compose(inference_tfm)

    for variant in data_variants:
        os.makedirs(os.path.join(log_dir, variant['tag']), exist_ok=True)
        logger.info(f"Start sampling for: {variant['tag']}")

        save_pdb(data_native, os.path.join(log_dir, variant['tag'], 'REF1.pdb'))       # w/  OpenMM minimization
    
        data_cropped = inference_tfm(
            copy.deepcopy(variant['data'])
        )
        data_cropped['template'] = inference_tfm(
            copy.deepcopy(variant['template'])
        )

        data_list_repeat = [ data_cropped ] * config.sampling.num_samples
        loader = DataLoader(data_list_repeat, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        batch_id = 0    #lll
        csv_batch_id = 0
        count = 0
        moe_debug = False
        moe_output_aa_csv = os.path.join(log_dir, variant['tag'], 'moe_cdr_output_aa_infer.csv')
        moe_routing_csv = os.path.join(log_dir, variant['tag'], 'moe_cdr_routing_infer.csv')
        moe_trace_csv = os.path.join(log_dir, variant['tag'], 'moe_cdr_routing_trace.csv')
        for batch in tqdm(loader, desc=variant['name'], dynamic_ncols=True):
            torch.set_grad_enabled(False)
            model.eval()
            batch = recursive_to(batch, args.device)
            if moe_debug:
                reset_moe_output_aa_stats(model)
                reset_moe_routing_stats(model)
            if 'abopt' in config.mode:
                # Antibody optimization starting from native
                traj_batch = model.optimize(batch, opt_step=variant['opt_step'], optimize_opt={
                    'pbar': True,
                    'sample_structure': config.sampling.sample_structure,
                    'sample_sequence': config.sampling.sample_sequence,
                })
            else:
                # De novo design    lll single=False,muti=True,
                traj_batch = model.sample(batch, batch_id, data_cropped, data_variant=variant['data'], log_dir=log_dir,
                                          tag=variant["tag"], sample_opt={
                    'pbar': True,
                    'sample_structure': config.sampling.sample_structure,
                    'sample_sequence': config.sampling.sample_sequence,
                    'single': config.sampling.single,
                    'multi': config.sampling.multi,
                    'scope': config.sampling.scope,
                    **get_energy_guidance_options(config),
                    'trace_moe': moe_debug,
                    'trace_moe_step': 99,
                    'trace_moe_csv': moe_trace_csv,
                    'trace_moe_max_tokens': 4096,
                })

            if moe_debug:
                moe_rows = flush_moe_output_aa_csv(model, moe_output_aa_csv, variant, csv_batch_id)
                if moe_rows > 0:
                    logger.info('[moe-output][infer] batch %04d | saved %d rows to %s' % (
                        csv_batch_id, moe_rows, moe_output_aa_csv
                    ))
                routing_rows = flush_moe_routing_csv(model, moe_routing_csv, variant, csv_batch_id)
                if routing_rows > 0:
                    logger.info('[moe-routing][infer] batch %04d | saved %d rows to %s' % (
                        csv_batch_id, routing_rows, moe_routing_csv
                    ))

            aa_new = traj_batch[0][2]   # 0: Last sampling step. 2: Amino acid.
            pos_atom_new, mask_atom_new = reconstruct_backbone_partially(
                pos_ctx = batch['pos_heavyatom'],
                R_new = so3vec_to_rotation(traj_batch[0][0]),
                t_new = traj_batch[0][1],
                aa = aa_new,
                chain_nb = batch['chain_nb'],
                res_nb = batch['res_nb'],
                mask_atoms = batch['mask_heavyatom'],
                mask_recons = batch['generate_flag'],
            )
            aa_new = aa_new.cpu()
            pos_atom_new = pos_atom_new.cpu()
            mask_atom_new = mask_atom_new.cpu()

            for i in range(aa_new.size(0)):
                data_tmpl = variant['data']
                aa = apply_patch_to_tensor(data_tmpl['aa'], aa_new[i], data_cropped['patch_idx'])
                mask_ha = apply_patch_to_tensor(data_tmpl['mask_heavyatom'], mask_atom_new[i], data_cropped['patch_idx'])
                pos_ha  = (
                    apply_patch_to_tensor(
                        data_tmpl['pos_heavyatom'], 
                        pos_atom_new[i] + batch['origin'][i].view(1, 1, 3).cpu(), 
                        data_cropped['patch_idx']
                    )
                )

                save_path = os.path.join(log_dir, variant['tag'], '%04d.pdb' % (count, ))
                save_pdb({
                    'chain_nb': data_tmpl['chain_nb'],
                    'chain_id': data_tmpl['chain_id'],
                    'resseq': data_tmpl['resseq'],
                    'icode': data_tmpl['icode'],
                    # Generated
                    'aa': aa,
                    'mask_heavyatom': mask_ha,
                    'pos_heavyatom': pos_ha,
                }, path=save_path)
                count += 1
            csv_batch_id += 1

        logger.info('Finished.\n')


if __name__ == '__main__':
    main()
