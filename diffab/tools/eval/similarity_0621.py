import numpy as np
from Bio.PDB import Selection
from Bio.PDB.Polypeptide import three_to_one
from Bio import pairwise2
from Bio.Align import substitution_matrices

from base import EvalTask, MULTI_CDR_REGION_DEFS


def reslist_rmsd(res_list1, res_list2):
    res_short, res_long = (res_list1, res_list2) if len(res_list1) < len(res_list2) else (res_list2, res_list1)
    M, N = len(res_short), len(res_long)

    def d(i, j):
        coord_i = np.array(res_short[i]['CA'].get_coord())
        coord_j = np.array(res_long[j]['CA'].get_coord())
        return ((coord_i - coord_j) ** 2).sum()

    SD = np.full([M, N], np.inf)
    for i in range(M):
        j = N - (M - i)
        SD[i, j] = sum([ d(i+k, j+k) for k in range(N-j) ])
    
    for j in range(N):
        SD[M-1, j] = d(M-1, j)

    for i in range(M-2, -1, -1):
        for j in range((N-(M-i))-1, -1, -1):
            SD[i, j] = min(
                d(i, j) + SD[i+1, j+1],
                SD[i, j+1]
            )

    min_SD = SD[0, :N-M+1].min()
    best_RMSD = np.sqrt(min_SD / M)
    return best_RMSD


def entity_to_seq(entity):
    seq = ''
    mapping = []
    for res in Selection.unfold_entities(entity, 'R'):
        try:
            seq += three_to_one(res.get_resname())
            mapping.append(res.get_id())
        except KeyError:
            pass
    assert len(seq) == len(mapping)
    return seq, mapping


def reslist_seqid(res_list1, res_list2):
    seq1, _ = entity_to_seq(res_list1)
    seq2, _ = entity_to_seq(res_list2)
    _, seq_id = align_sequences(seq1, seq2)
    return seq_id


def align_sequences(sequence_A, sequence_B, **kwargs):
    """
    Performs a global pairwise alignment between two sequences
    using the BLOSUM62 matrix and the Needleman-Wunsch algorithm
    as implemented in Biopython. Returns the alignment, the sequence
    identity and the residue mapping between both original sequences.
    """

    def _calculate_identity(sequenceA, sequenceB):
        """
        Returns the percentage of identical characters between two sequences.
        Assumes the sequences are aligned.
        """

        sa, sb, sl = sequenceA, sequenceB, len(sequenceA)
        matches = [sa[i] == sb[i] for i in range(sl)]
        seq_id = (100 * sum(matches)) / sl
        return seq_id

        # gapless_sl = sum([1 for i in range(sl) if (sa[i] != '-' and sb[i] != '-')])
        # gap_id = (100 * sum(matches)) / gapless_sl
        # return (seq_id, gap_id)

    #
    matrix = kwargs.get('matrix', substitution_matrices.load("BLOSUM62"))
    gap_open = kwargs.get('gap_open', -10.0)
    gap_extend = kwargs.get('gap_extend', -0.5)

    """
    alns = pairwise2.align.globalds(sequence_A, sequence_B,
                                    matrix, gap_open, gap_extend,
                                    penalize_end_gaps=(False, False) )

    best_aln = alns[0]
    aligned_A, aligned_B, score, begin, end = best_aln
    # Calculate sequence identity
    seq_id = _calculate_identity(aligned_A, aligned_B)
    return (aligned_A, aligned_B), seq_id
    """
    seq_id = _calculate_identity(sequence_A, sequence_B)
    return (), seq_id


def extract_reslist(model, residue_first, residue_last):
    assert residue_first[0] == residue_last[0]
    residue_first, residue_last = tuple(residue_first), tuple(residue_last)

    chain_id = residue_first[0]
    pos_first, pos_last = residue_first[1:], residue_last[1:]
    chain = model[chain_id]
    reslist = []
    for res in Selection.unfold_entities(chain, 'R'):
        pos_current = (res.id[1], res.id[2])
        if pos_first <= pos_current <= pos_last:
            reslist.append(res)
    return reslist


def _normalize_cdr_name(name):
    if name.startswith('H_CDR'):
        return f"H{name[-1]}"
    if name.startswith('L_CDR'):
        return f"L{name[-1]}"
    return name


def _iter_multicdr_regions(task: EvalTask):
    selected_cdrs = [_normalize_cdr_name(cdr) for cdr in task.info.get('cdrs', [])]
    if not selected_cdrs:
        return []
    selected_cdrs = set(selected_cdrs)

    cdrs = [name for name, _, _, _ in MULTI_CDR_REGION_DEFS if name in selected_cdrs]

    region_count = len(cdrs)
    starts = task.residue_first[:region_count]
    ends = task.residue_last[:region_count]
    trailing = task.residue_first[region_count:]

    trailing_idx = 0
    chain_ids = {}
    if any(cdr.startswith('H') for cdr in cdrs):
        chain_ids['H'] = trailing[trailing_idx]
        trailing_idx += 1
    if any(cdr.startswith('L') for cdr in cdrs):
        chain_ids['L'] = trailing[trailing_idx]
        trailing_idx += 1

    regions = []
    for cdr_name, start, end in zip(cdrs, starts, ends):
        chain_prefix = cdr_name[0]
        chain_id = chain_ids.get(chain_prefix)
        if chain_id is None:
            continue
        regions.append((cdr_name, int(start), int(end), chain_id))
    return regions


# def eval_similarity(task: EvalTask):
#     model_gen = task.get_gen_biopython_model()
#     model_ref = task.get_ref_biopython_model()
#     print(task.residue_first, task.residue_last)    #[95, 52, 26, 89, 50, 24, 'C', 'B', 'A'] [102, 56, 32, 97, 56, 34, 'C', 'B', 'A']
#     # assert False,'debug'
#     reslist_gen = extract_reslist(model_gen, task.residue_first, task.residue_last)
#     reslist_ref = extract_reslist(model_ref, task.residue_first, task.residue_last)

#     task.scores.update({
#         'rmsd': reslist_rmsd(reslist_gen, reslist_ref),
#         'seqid': reslist_seqid(reslist_gen, reslist_ref),
#     })
#     return task

def eval_similarity(task: EvalTask):
    model_gen = task.get_gen_biopython_model()
    model_ref = task.get_ref_biopython_model()
    
    # # 打印输入以便调试确认
    # print(f"Eval Input: {task.residue_first} - {task.residue_last}")

    # 判断是否为多区域列表模式
    # 依据是：输入是列表且长度足以包含CDR信息和链信息 (示例长度为9)
    if isinstance(task.residue_first, list) and task.info.get('tag') == 'MultipleCDRs':
        region_map = _iter_multicdr_regions(task)
        scores_update = {}

        for name, r_start_num, r_end_num, chain_id in region_map:
            try:
                target_first = (chain_id, r_start_num, ' ')
                target_last  = (chain_id, r_end_num, ' ')

                reslist_gen = extract_reslist(model_gen, target_first, target_last)
                reslist_ref = extract_reslist(model_ref, target_first, target_last)

                if len(reslist_gen) > 0 and len(reslist_ref) > 0:
                    val_rmsd = reslist_rmsd(reslist_gen, reslist_ref)
                    val_seqid = reslist_seqid(reslist_gen, reslist_ref)
                else:
                    print(f"Warning: Empty reslist for {name} chain {chain_id} {r_start_num}-{r_end_num}")
                    val_rmsd = -1.0 
                    val_seqid = 0.0

                scores_update[f'rmsd_{name}'] = val_rmsd
                scores_update[f'seqid_{name}'] = val_seqid

            except Exception as e:
                print(f"Error calculating {name}: {e}")
                scores_update[f'rmsd_{name}'] = -1.0
                scores_update[f'seqid_{name}'] = 0.0

        task.scores.update(scores_update)

    else:
        # === 原有逻辑：处理单个连续区域 ===
        # 保持此逻辑以兼容非列表输入的旧任务
        reslist_gen = extract_reslist(model_gen, task.residue_first, task.residue_last)
        reslist_ref = extract_reslist(model_ref, task.residue_first, task.residue_last)

        task.scores.update({
            'rmsd': reslist_rmsd(reslist_gen, reslist_ref),
            'seqid': reslist_seqid(reslist_gen, reslist_ref),
        })
    
    return task
