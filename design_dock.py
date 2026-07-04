import os
import shutil
import argparse
from diffab.tools.dock.hdock import HDockAntibody, DockSite
from diffab.tools.runner.design_for_pdb import args_factory, design_for_pdb


def parse_epitope_sites(values):
    if values is None:
        return None
    sites = []
    for value in values:
        if ':' not in value:
            raise ValueError(f'Invalid epitope site format: {value}. Expected CHAIN:RESSEQ, for example A:991.')
        chain, resseq_str = value.split(':', 1)
        if not chain:
            raise ValueError(f'Invalid epitope site format: {value}. Chain id is empty.')
        try:
            resseq = int(resseq_str)
        except ValueError as exc:
            raise ValueError(f'Invalid epitope site format: {value}. Residue number must be an integer.') from exc
        sites.append(DockSite(chain, resseq))
    return sites


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--antigen', type=str, required=True)
    parser.add_argument('--antibody', type=str, default='./data/examples/3QHF_Fv.pdb')
    parser.add_argument('--heavy', type=str, default='H', help='Chain id of the heavy chain.')
    parser.add_argument('--light', type=str, default='L', help='Chain id of the light chain.')
    parser.add_argument('--hdock_bin', type=str, default='./bin/hdock')
    parser.add_argument('--createpl_bin', type=str, default='./bin/createpl')
    parser.add_argument(
        '--cdrs',
        nargs='+',
        default=['H3'],
        choices=['H1', 'H2', 'H3', 'L1', 'L2', 'L3'],
        help='Selected CDR regions for HDOCK. Use one or more from H1 H2 H3 L1 L2 L3.'
    )
    parser.add_argument(
        '--epitope_sites',
        nargs='+',
        default=None,
        help='Selected antigen sites for HDOCK, formatted as CHAIN:RESSEQ, for example A:991 A:992.'
    )
    parser.add_argument('-n', '--num_docks', type=int, default=10)
    parser.add_argument('-c', '--config', type=str, default='./configs/test/codesign_single.yml')
    parser.add_argument(
        '--template', '--template_dict',
        dest='template',
        type=str,
        default=None,
        help='Template pickle path or template dict. Kept for backward compatibility; ignored by the current design_for_pdb.'
    )
    parser.add_argument('-o', '--out_root', type=str, default='./results')
    parser.add_argument('-t', '--tag', type=str, default='')
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    args = parser.parse_args()

    epitope_sites = parse_epitope_sites(args.epitope_sites)

    hdock_missing = []
    if not os.path.exists(args.hdock_bin):
        hdock_missing.append(args.hdock_bin)
    if not os.path.exists(args.createpl_bin):
        hdock_missing.append(args.createpl_bin)
    if len(hdock_missing) > 0:
        print("[WARNING] The following HDOCK applications are missing:")
        for f in hdock_missing:
            print(f" > {f}")
        print("Please download HDOCK from http://huanglab.phys.hust.edu.cn/software/hdocklite/ "
                "and put `hdock` and `createpl` to the above path.")
        exit()

    antigen_name = os.path.basename(os.path.splitext(args.antigen)[0])
    docked_pdb_dir = os.path.join(os.path.splitext(args.antigen)[0] + '_dock')
    docked_meta_dir = os.path.join(docked_pdb_dir, 'out')
    os.makedirs(docked_pdb_dir, exist_ok=True)
    for fname in os.listdir(docked_pdb_dir):  # 411
        if fname.endswith('.pdb') and '_chothia' in fname:  # 411
            stale_path = os.path.join(docked_pdb_dir, fname)  # 411
            os.remove(stale_path)  # 411
            print(f'[INFO] Remove stale intermediate: {stale_path}')  # 411

    docked_pdb_paths = []
    for fname in os.listdir(docked_pdb_dir):
        if fname.endswith('.pdb') and '_chothia' not in fname:  # 411
            docked_pdb_paths.append(os.path.join(docked_pdb_dir, fname))
    docked_pdb_paths.sort()
    if len(docked_pdb_paths) < args.num_docks:
        missing_docks = args.num_docks - len(docked_pdb_paths)
        print(f'[INFO] Reuse {len(docked_pdb_paths)} existing docking poses, generate {missing_docks} more')
        with HDockAntibody(cdrs=args.cdrs) as dock_session:
            dock_session.set_antigen(args.antigen, epitope_sites=epitope_sites)
            dock_session.set_antibody(args.antibody)
            docked_tmp_paths = dock_session.dock(nmax=args.num_docks)
            generated_count = len(docked_tmp_paths)
            print(f'[INFO] HDOCK returned {generated_count} docking poses in this run')
            copied_pdb_paths = []
            start_idx = len(docked_pdb_paths)
            new_tmp_paths = docked_tmp_paths[:missing_docks]
            if generated_count < missing_docks:
                print(f'[WARNING] Requested {missing_docks} new poses, but HDOCK only returned {generated_count}')
            for i, tmp_path in enumerate(new_tmp_paths, start=start_idx):
                dest_path = os.path.join(docked_pdb_dir, f"{antigen_name}_Ab_{i:04d}.pdb")
                shutil.copyfile(tmp_path, dest_path)
                print(f'[INFO] Copy {tmp_path} -> {dest_path}')
                docked_pdb_paths.append(dest_path)
                copied_pdb_paths.append(dest_path)
            if copied_pdb_paths:
                dock_session.export_last_outputs(docked_meta_dir, copied_pdb_paths, start_rank=start_idx + 1)
                print(f'[INFO] Export HDOCK outputs to {docked_meta_dir}')

    for pdb_path in docked_pdb_paths:
        current_args = dict(vars(args))  # 411
        base_tag = current_args.get('tag', '')  # 411
        current_args['tag'] = f"{base_tag}{antigen_name}"  # 411
        design_args = args_factory(
            pdb_path = pdb_path,
            **current_args,
        )
        design_for_pdb(design_args)


if __name__ == '__main__':
    main()
