# pyright: reportMissingImports=false
import os
import time
import pyrosetta
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task import operation
from pyrosetta.rosetta.core.select import residue_selector as selections
from pyrosetta.rosetta.core.select.movemap import MoveMapFactory, move_map_action
import logging
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
    from .base import RelaxTask
except ImportError:
    from base import RelaxTask


def current_milli_time():
    return round(time.time() * 1000)


def parse_residue_position(p):
    icode = None
    if not p[-1].isnumeric():   # Has ICODE
        icode = p[-1]

    for i, c in enumerate(p):
        if c.isnumeric():
            break
    chain = p[:i]
    resseq = int(p[i:])

    if icode is not None:
        return chain, resseq, icode
    else:
        return chain, resseq


def get_scorefxn(scorefxn_name:str):
    """
    Gets the scorefxn with appropriate corrections.
    Taken from: https://gist.github.com/matteoferla/b33585f3aeab58b8424581279e032550
    """
    import pyrosetta

    corrections = {
        'beta_july15': False,
        'beta_nov16': False,
        'gen_potential': False,
        'restore_talaris_behavior': False,
    }
    if 'beta_july15' in scorefxn_name or 'beta_nov15' in scorefxn_name:
        # beta_july15 is ref2015
        corrections['beta_july15'] = True
    elif 'beta_nov16' in scorefxn_name:
        corrections['beta_nov16'] = True
    elif 'genpot' in scorefxn_name:
        corrections['gen_potential'] = True
        pyrosetta.rosetta.basic.options.set_boolean_option('corrections:beta_july15', True)
    elif 'talaris' in scorefxn_name:  #2013 and 2014
        corrections['restore_talaris_behavior'] = True
    else:
        pass
    for corr, value in corrections.items():
        pyrosetta.rosetta.basic.options.set_boolean_option(f'corrections:{corr}', value)
    return pyrosetta.create_score_function(scorefxn_name)


# class RelaxRegion(object):
    
#     def __init__(self, scorefxn='ref2015', max_iter=1000, subset='nbrs', move_bb=True):
#         super().__init__()
#         self.scorefxn = get_scorefxn(scorefxn)
#         self.fast_relax = FastRelax()
#         self.fast_relax.set_scorefxn(self.scorefxn)
#         self.fast_relax.max_iter(max_iter)
#         assert subset in ('all', 'target', 'nbrs')
#         self.subset = subset
#         self.move_bb = move_bb

#     def __call__(self, pdb_path, flexible_residue_first, flexible_residue_last):
#         print(pdb_path)
#         try:
#             pose = pyrosetta.pose_from_pdb(pdb_path)
#         except:
#             logging.warning(f'{pdb_path}')
#             if not pdb_path:
#                 raise TypeError
#             else:
#                 raise RuntimeError
#         start_t = current_milli_time()
#         original_pose = pose.clone()

#         tf = TaskFactory()
#         tf.push_back(operation.InitializeFromCommandline())
#         tf.push_back(operation.RestrictToRepacking())   # Only allow residues to repack. No design at any position.

#         # Create selector for the region to be relaxed
#         # Turn off design and repacking on irrelevant positions
#         if flexible_residue_first[-1] == ' ': 
#             flexible_residue_first = flexible_residue_first[:-1]
#         if flexible_residue_last[-1] == ' ':  
#             flexible_residue_last  = flexible_residue_last[:-1]
#         if self.subset != 'all':
#             gen_selector = selections.ResidueIndexSelector()
#             gen_selector.set_index_range(
#                 pose.pdb_info().pdb2pose(*flexible_residue_first), 
#                 pose.pdb_info().pdb2pose(*flexible_residue_last), 
#             )
#             nbr_selector = selections.NeighborhoodResidueSelector()
#             nbr_selector.set_focus_selector(gen_selector)
#             nbr_selector.set_include_focus_in_subset(True)

#             if self.subset == 'nbrs':
#                 subset_selector = nbr_selector
#             elif self.subset == 'target':
#                 subset_selector = gen_selector

#             prevent_repacking_rlt = operation.PreventRepackingRLT()
#             prevent_subset_repacking = operation.OperateOnResidueSubset(
#                 prevent_repacking_rlt, 
#                 subset_selector,
#                 flip_subset=True,
#             )
#             tf.push_back(prevent_subset_repacking)

#         scorefxn = self.scorefxn
#         fr = self.fast_relax

#         pose = original_pose.clone()
#         pos_list = pyrosetta.rosetta.utility.vector1_unsigned_long()
#         for pos in range(pose.pdb_info().pdb2pose(*flexible_residue_first), pose.pdb_info().pdb2pose(*flexible_residue_last)+1):
#             pos_list.append(pos)
#         # basic_idealize(pose, pos_list, scorefxn, fast=True)

#         mmf = MoveMapFactory()
#         if self.move_bb: 
#             mmf.add_bb_action(move_map_action.mm_enable, gen_selector)
#         mmf.add_chi_action(move_map_action.mm_enable, subset_selector)
#         mm  = mmf.create_movemap_from_pose(pose)

#         fr.set_movemap(mm)
#         fr.set_task_factory(tf)
#         fr.apply(pose)

#         e_before = scorefxn(original_pose)
#         e_relax  = scorefxn(pose) 
#         # print('\n\n[Finished in %.2f secs]' % ((current_milli_time() - start_t) / 1000))
#         # print(' > Energy (before):    %.4f' % scorefxn(original_pose))
#         # print(' > Energy (optimized): %.4f' % scorefxn(pose))
#         return pose, e_before, e_relax



class RelaxRegion(object):
    
    def __init__(self, scorefxn='ref2015', max_iter=1000, subset='nbrs', move_bb=True):
        super().__init__()
        self.scorefxn = get_scorefxn(scorefxn)
        self.fast_relax = FastRelax()
        self.fast_relax.set_scorefxn(self.scorefxn)
        self.fast_relax.max_iter(max_iter)
        assert subset in ('all', 'target', 'nbrs')
        self.subset = subset
        self.move_bb = move_bb

    def __call__(self, pdb_path, flexible_residue_first, flexible_residue_last, cdrs=None):
        # print(pdb_path)
        try:
            pose = pyrosetta.pose_from_pdb(pdb_path)
        except:
            logging.warning(f'{pdb_path}')
            if not pdb_path:
                raise TypeError
            else:
                raise RuntimeError
        start_t = current_milli_time()
        original_pose = pose.clone()

        tf = TaskFactory()
        tf.push_back(operation.InitializeFromCommandline())
        tf.push_back(operation.RestrictToRepacking())   # Only allow residues to repack. No design at any position.

        # 判断是否为多区域列表模式
        # 列表模式特征: 长度大于2 (因为包含 H1-H3, L1-L3 和 Chain IDs)
        is_multi_region = isinstance(flexible_residue_first, list) and len(flexible_residue_first) > 3

        if not is_multi_region:
            # === 原有逻辑: 单一区域处理 ===
            if flexible_residue_first[-1] == ' ': 
                flexible_residue_first = flexible_residue_first[:-1]
            if flexible_residue_last[-1] == ' ':  
                flexible_residue_last  = flexible_residue_last[:-1]
        
        gen_selector = selections.ResidueIndexSelector()
        
        if self.subset != 'all':
            if is_multi_region:
                normalized_cdrs = []
                for cdr in cdrs or []:
                    if cdr.startswith('H_CDR'):
                        normalized_cdrs.append(f"H{cdr[-1]}")
                    elif cdr.startswith('L_CDR'):
                        normalized_cdrs.append(f"L{cdr[-1]}")
                    else:
                        normalized_cdrs.append(cdr)

                if not normalized_cdrs:
                    normalized_cdrs = ['H3', 'H2', 'H1', 'L3', 'L2', 'L1']

                region_count = len(normalized_cdrs)
                trailing = flexible_residue_first[region_count:]
                trailing_idx = 0
                chain_ids = {}
                if any(cdr.startswith('H') for cdr in normalized_cdrs):
                    chain_ids['H'] = trailing[trailing_idx]
                    trailing_idx += 1
                if any(cdr.startswith('L') for cdr in normalized_cdrs):
                    chain_ids['L'] = trailing[trailing_idx]
                    trailing_idx += 1
                
                pose_indices_str = ""

                for i, cdr_name in enumerate(normalized_cdrs):
                    chain_id = chain_ids.get(cdr_name[0])
                    if chain_id is None:
                        continue
                    start_pdb = int(flexible_residue_first[i])
                    end_pdb = int(flexible_residue_last[i])
                    start_pose = pose.pdb_info().pdb2pose(chain_id, start_pdb)
                    end_pose = pose.pdb_info().pdb2pose(chain_id, end_pdb)

                    if start_pose != 0 and end_pose != 0:
                        pose_indices_str += f"{start_pose}-{end_pose},"
                    else:
                        logging.warning(f"{cdr_name} region {start_pdb}-{end_pdb} on chain {chain_id} not found in pose.")

                # 应用所有区域到 Selector
                if pose_indices_str:
                    gen_selector.set_index(pose_indices_str.strip(','))
                
            else:
                # === 原有逻辑: 设置单个区域范围 ===
                gen_selector.set_index_range(
                    pose.pdb_info().pdb2pose(*flexible_residue_first), 
                    pose.pdb_info().pdb2pose(*flexible_residue_last), 
                )

            # 下面的逻辑保持不变，基于 gen_selector 创建 subset
            nbr_selector = selections.NeighborhoodResidueSelector()
            nbr_selector.set_focus_selector(gen_selector)
            nbr_selector.set_include_focus_in_subset(True)

            if self.subset == 'nbrs':
                subset_selector = nbr_selector
            elif self.subset == 'target':
                subset_selector = gen_selector

            prevent_repacking_rlt = operation.PreventRepackingRLT()
            prevent_subset_repacking = operation.OperateOnResidueSubset(
                prevent_repacking_rlt, 
                subset_selector,
                flip_subset=True,
            )
            tf.push_back(prevent_subset_repacking)

        scorefxn = self.scorefxn
        fr = self.fast_relax

        pose = original_pose.clone()
        
        # 这里的 pos_list 逻辑主要是为了 basic_idealize (虽然被注释掉了)，
        # 为了代码完整性，如果用户需要 future use，也可以在这里根据 gen_selector 更新 pos_list
        # 但为了保持"现有代码逻辑"不变，我们只针对 gen_selector 做了修改，MoveMap 会自动处理选中的区域。
        
        mmf = MoveMapFactory()
        if self.move_bb: 
            # gen_selector 现在包含了所有 CDR 区域
            mmf.add_bb_action(move_map_action.mm_enable, gen_selector)
        
        # subset_selector (如果是 nbrs) 现在包含所有 CDR 及其邻居
        mmf.add_chi_action(move_map_action.mm_enable, subset_selector)
        mm  = mmf.create_movemap_from_pose(pose)

        fr.set_movemap(mm)
        fr.set_task_factory(tf)
        fr.apply(pose)

        e_before = scorefxn(original_pose)
        e_relax  = scorefxn(pose) 
        # print('\n\n[Finished in %.2f secs]' % ((current_milli_time() - start_t) / 1000))
        # print(' > Energy (before):    %.4f' % scorefxn(original_pose))
        # print(' > Energy (optimized): %.4f' % scorefxn(pose))
        return pose, e_before, e_relax

def run_pyrosetta(task: RelaxTask):
    if not task.can_proceed() :
        return task
    if task.update_if_finished('rosetta'):
        return task

    
    # print(task.flexible_residue_first)
    # print(task.flexible_residue_last)
    # assert False,'debug'
    minimizer = RelaxRegion()
    pose_min, _, _ = minimizer(
        pdb_path = task.current_path,
        flexible_residue_first = task.flexible_residue_first,
        flexible_residue_last = task.flexible_residue_last,
        cdrs = task.info.get('cdrs'),
    )

    out_path = task.set_current_path_tag('rosetta')
    pose_min.dump_pdb(out_path)
    task.mark_success()
    return task


def run_pyrosetta_fixbb(task: RelaxTask):
    if not task.can_proceed() :
        return task
    if task.update_if_finished('fixbb'):
        return task

    minimizer = RelaxRegion(move_bb=False)
    pose_min, _, _ = minimizer(
        pdb_path = task.current_path,
        flexible_residue_first = task.flexible_residue_first,
        flexible_residue_last = task.flexible_residue_last,
        cdrs = task.info.get('cdrs'),
    )

    out_path = task.set_current_path_tag('fixbb')
    pose_min.dump_pdb(out_path)
    task.mark_success()
    return task

    
