# pyright: reportMissingImports=false
import pyrosetta
import logging
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
pyrosetta.init(' '.join([
    '-mute', 'all',
    '-use_input_sc',
    '-ignore_unrecognized_res',
    '-ignore_zero_occupancy', 'false',
    '-load_PDB_components', 'false',
    '-relax:default_repeats', '2',
    '-no_fconfig',
]))

try:
    from .base import EvalTask
except ImportError:
    from base import EvalTask


def pyrosetta_interface_energy(pdb_path, interface):
    pose = pyrosetta.pose_from_pdb(pdb_path)
    mover = InterfaceAnalyzerMover(interface)
    mover.set_pack_separated(True)
    mover.apply(pose)
    return pose.scores['dG_separated']


def eval_interface_energy(task: EvalTask):
    model_gen = task.get_gen_biopython_model()
    antigen_chains = set()
    for chain in model_gen:
        if chain.id not in task.ab_chains:
            antigen_chains.add(chain.id)
    antigen_chains = ''.join(sorted(antigen_chains))
    antibody_chains = ''.join(task.ab_chains)
    if not antibody_chains or not antigen_chains:
        logging.warning(
            'Skipping interface energy for %s because antibody_chains=%r antigen_chains=%r',
            task.in_path, antibody_chains, antigen_chains
        )
        task.scores.update({
            'dG_gen': float('nan'),
            'dG_ref': float('nan'),
            'ddG': float('nan'),
        })
        return task
    interface = f"{antibody_chains}_{antigen_chains}"

    dG_gen = pyrosetta_interface_energy(task.in_path, interface)
    dG_ref = pyrosetta_interface_energy(task.ref_path, interface)

    task.scores.update({
        'dG_gen': dG_gen,
        'dG_ref': dG_ref,
        'ddG': dG_gen - dG_ref
    })
    return task
