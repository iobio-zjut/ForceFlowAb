import argparse
import ray
import time

from openmm_relaxer import run_openmm
from pyrosetta_relaxer import run_pyrosetta, run_pyrosetta_fixbb
from base import TaskScanner


def pipeline_openmm_pyrosetta(task):
    # 523: Keep the pipeline logic local to one Ray worker to avoid nested remote submission.
    # Previous nested version for reference:
    # funcs = [
    #     run_openmm_remote,
    #     run_pyrosetta_remote,
    # ]
    # for fn in funcs:
    #     task = fn.remote(task)
    # return ray.get(task)
    task = run_openmm(task)
    task = run_pyrosetta(task)
    return task


def pipeline_pyrosetta(task):
    # 523: Run PyRosetta directly inside the assigned worker instead of spawning another Ray task.
    # Previous nested version for reference:
    # funcs = [
    #     run_pyrosetta_remote,
    # ]
    # for fn in funcs:
    #     task = fn.remote(task)
    # return ray.get(task)
    return run_pyrosetta(task)


def pipeline_pyrosetta_fixbb(task):
    # 523: Same flattening for fixbb; the old implementation nested a second remote task here.
    # Previous nested version for reference:
    # funcs = [
    #     run_pyrosetta_fixbb_remote,
    # ]
    # for fn in funcs:
    #     task = fn.remote(task)
    # return ray.get(task)
    return run_pyrosetta_fixbb(task)


@ray.remote(num_gpus=1/8, num_cpus=1)
def pipeline_openmm_pyrosetta_remote(task):
    # 523: Single remote boundary for the full openmm->pyrosetta pipeline.
    return pipeline_openmm_pyrosetta(task)


@ray.remote(num_cpus=1)
def pipeline_pyrosetta_remote(task):
    # 523: Single remote boundary for the PyRosetta-only pipeline.
    return pipeline_pyrosetta(task)


@ray.remote(num_cpus=1)
def pipeline_pyrosetta_fixbb_remote(task):
    # 523: Single remote boundary for the fixbb pipeline.
    return pipeline_pyrosetta_fixbb(task)


pipeline_dict = {
    'openmm_pyrosetta': pipeline_openmm_pyrosetta_remote,
    'pyrosetta': pipeline_pyrosetta_remote,
    'pyrosetta_fixbb': pipeline_pyrosetta_fixbb_remote,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./results')
    parser.add_argument('--pipeline', type=lambda s: pipeline_dict[s], default=pipeline_openmm_pyrosetta)
    parser.add_argument('--num-cpus', type=int, default=15)
    args = parser.parse_args()
    ray.init(num_cpus=args.num_cpus, include_dashboard=False)
    print('start')

    final_pfx = 'fixbb' if args.pipeline == pipeline_pyrosetta_fixbb_remote else 'rosetta'
    scanner = TaskScanner(args.root, final_postfix=final_pfx)
    # while True:
    # print(1)
    tasks = scanner.scan()
    # print(2)
    futures = [args.pipeline.remote(t) for t in tasks]
    # print(3)
    if len(futures) > 0:
        print(f'Submitted {len(futures)} tasks.')
    while len(futures) > 0:
        done_ids, futures = ray.wait(futures, num_returns=1)
        for done_id in done_ids:
            done_task = ray.get(done_id)
            print(f'Remaining {len(futures)}. Finished {done_task.current_path}')
    time.sleep(1.0)

if __name__ == '__main__':
    main()
