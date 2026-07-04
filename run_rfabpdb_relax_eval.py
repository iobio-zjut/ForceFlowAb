#!/usr/bin/env python
import os
import sys
#运行这个来把/xsdata/lzhlzh/26_36/FlowAB/FlowDesign/results/rfabpdb下面的pdb文件都relax一下，并且计算interface dG，结果写到rfabpdb_relaxed_dg.csv里面
# Keep the native libraries from oversubscribing threads inside each Ray worker.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
RELAX_TOOL_DIR = os.path.join(PROJECT_ROOT, 'diffab', 'tools', 'relax')
if RELAX_TOOL_DIR not in sys.path:
    sys.path.insert(0, RELAX_TOOL_DIR)

import argparse
import csv
from dataclasses import dataclass
from typing import List

import ray
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

from diffab.tools.relax.pyrosetta_relaxer import RelaxRegion
import pyrosetta


DEFAULT_RESIDUE_FIRST = [95, 52, 26, 'H', 'T']
DEFAULT_RESIDUE_LAST = [102, 56, 32, 'H', 'T']
DEFAULT_CDRS = ['H_CDR3', 'H_CDR2', 'H_CDR1']
DEFAULT_INTERFACE = 'H_T'


@dataclass
class RelaxEvalResult:
    filename: str
    input_path: str
    relaxed_path: str
    status: str
    dG: float
    error: str = ''


def list_input_pdbs(root: str) -> List[str]:
    pdbs = []
    for name in sorted(os.listdir(root)):
        if not name.endswith('.pdb'):
            continue
        if name.endswith('_rosetta.pdb'):
            continue
        path = os.path.join(root, name)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            pdbs.append(path)
    return pdbs


def get_relaxed_path(pdb_path: str) -> str:
    stem, ext = os.path.splitext(pdb_path)
    return f'{stem}_rosetta{ext}'


def compute_interface_dg(pdb_path: str, interface: str) -> float:
    pose = pyrosetta.pose_from_pdb(pdb_path)
    mover = InterfaceAnalyzerMover(interface)
    mover.set_pack_separated(True)
    mover.apply(pose)
    return pose.scores['dG_separated']


def relax_one_pdb(pdb_path: str, residue_first, residue_last, cdrs) -> str:
    minimizer = RelaxRegion()
    pose_min, _, _ = minimizer(
        pdb_path=pdb_path,
        flexible_residue_first=residue_first,
        flexible_residue_last=residue_last,
        cdrs=cdrs,
    )
    out_path = get_relaxed_path(pdb_path)
    pose_min.dump_pdb(out_path)
    return out_path


@ray.remote(num_cpus=1)
def process_pdb_remote(pdb_path: str, residue_first, residue_last, cdrs, interface: str):
    try:
        relaxed_path = get_relaxed_path(pdb_path)
        if not (os.path.exists(relaxed_path) and os.path.getsize(relaxed_path) > 0):
            relaxed_path = relax_one_pdb(pdb_path, residue_first, residue_last, cdrs)
        dG = compute_interface_dg(relaxed_path, interface)
        return RelaxEvalResult(
            filename=os.path.basename(pdb_path),
            input_path=pdb_path,
            relaxed_path=relaxed_path,
            status='success',
            dG=dG,
        )
    except Exception as exc:
        return RelaxEvalResult(
            filename=os.path.basename(pdb_path),
            input_path=pdb_path,
            relaxed_path=get_relaxed_path(pdb_path),
            status='failed',
            dG=float('nan'),
            error=str(exc),
        )


def write_csv(results: List[RelaxEvalResult], out_csv: str):
    fieldnames = ['filename', 'input_path', 'relaxed_path', 'status', 'dG', 'error']
    with open(out_csv, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow({
                'filename': item.filename,
                'input_path': item.input_path,
                'relaxed_path': item.relaxed_path,
                'status': item.status,
                'dG': item.dG,
                'error': item.error,
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root',
        type=str,
        default='/xsdata/lzhlzh/26_36/FlowAB/FlowDesign/results/rfabpdb',
    )
    parser.add_argument(
        '--out-csv',
        type=str,
        default='/xsdata/lzhlzh/26_36/FlowAB/FlowDesign/results/rfabpdb/rfabpdb_relaxed_dg.csv',
    )
    parser.add_argument('--num-cpus', type=int, default=30)
    args = parser.parse_args()

    pdbs = list_input_pdbs(args.root)
    if not pdbs:
        raise SystemExit(f'No input PDB files found under {args.root}')

    ray.init(num_cpus=args.num_cpus, include_dashboard=False)
    futures = [
        process_pdb_remote.remote(
            pdb_path,
            DEFAULT_RESIDUE_FIRST,
            DEFAULT_RESIDUE_LAST,
            DEFAULT_CDRS,
            DEFAULT_INTERFACE,
        )
        for pdb_path in pdbs
    ]

    results = []
    while futures:
        done_ids, futures = ray.wait(futures, num_returns=1)
        for done_id in done_ids:
            result = ray.get(done_id)
            results.append(result)
            print(f'Finished {result.filename} status={result.status}')

    ray.shutdown()
    results.sort(key=lambda x: x.filename)
    write_csv(results, args.out_csv)
    print(f'Wrote {len(results)} rows to {args.out_csv}')


if __name__ == '__main__':
    main()
