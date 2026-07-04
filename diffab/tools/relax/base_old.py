import os
import re
import json
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class RelaxTask:
    in_path: str
    current_path: str
    info: dict
    status: str

    flexible_residue_first: Optional[Tuple] = None
    flexible_residue_last: Optional[Tuple] = None
    H_chain:  Optional[str]=None
    L_chain:  Optional[str]=None
    A_chain:  Optional[str]=None

    def get_in_path_with_tag(self, tag):
        name, ext = os.path.splitext(self.in_path)
        new_path = f'{name}_{tag}{ext}'
        return new_path

    def set_current_path_tag(self, tag):
        new_path = self.get_in_path_with_tag(tag)
        self.current_path = new_path
        return new_path

    def check_current_path_exists(self):
        ok = os.path.exists(self.current_path)
        if not ok:
            self.mark_failure()
        if os.path.getsize(self.current_path) == 0:
            ok = False
            self.mark_failure()
            os.unlink(self.current_path)
        return ok

    def update_if_finished(self, tag):
        out_path = self.get_in_path_with_tag(tag)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            # print('Already finished', out_path)
            self.set_current_path_tag(tag)
            self.mark_success()
            return True
        return False

    def can_proceed(self):
        self.check_current_path_exists()
        return self.status != 'failed'

    def mark_success(self):
        self.status = 'success'

    def mark_failure(self):
        self.status = 'failed'



class TaskScanner:

    def __init__(self, root, final_postfix=None):
        super().__init__()
        self.root = root
        self.visited = set()
        self.final_postfix = final_postfix

    def _get_metadata(self, fpath):
        json_path = os.path.join(
            os.path.dirname(os.path.dirname(fpath)), 
            'metadata.json'
        )
        tag_name = os.path.basename(os.path.dirname(fpath))
        try:
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            for item in metadata['items']:
                if item['tag'] == tag_name:
                    return item
        except (json.JSONDecodeError, FileNotFoundError) as e:
            return None
        return None

    def scan(self) -> List[RelaxTask]: 
        tasks = []
        input_fname_pattern = '(^\d+\.pdb$|^REF\d\.pdb$|^P.*\.pdb$)'
        for parent, _, files in os.walk(self.root):
            for fname in files:
                fpath = os.path.join(parent, fname)
                if not re.match(input_fname_pattern, fname):
                    continue
                if os.path.getsize(fpath) == 0:
                    continue
                if fpath in self.visited:
                    continue
                
                # If finished
                if self.final_postfix is not None:
                    fpath_name, fpath_ext = os.path.splitext(fpath)
                    fpath_final = f"{fpath_name}_{self.final_postfix}{fpath_ext}"
                    if os.path.exists(fpath_final):
                        continue

                # Get metadata
                info = self._get_metadata(fpath)
                if info is None:
                    continue
                    
                ##LZH Changed
                if info.get('tag')=='MultipleCDRs':
                    info['residue_first'] = [95,52,26,89,50,24]
                    info['residue_last'] = [102,56,32,97,56,34]
                    pname=info.get('name')
                    H_chain = pname.split('_')[1]
                    L_chain = pname.split('_')[2]
                    A_chain = pname.split('_')[3].split('-')[0]
                    info['residue_first'].append(H_chain)
                    info['residue_first'].append(L_chain)
                    info['residue_first'].append(A_chain)
                    info['residue_last'].append(H_chain)
                    info['residue_last'].append(L_chain)
                    info['residue_last'].append(A_chain)
                ##
                tasks.append(RelaxTask(
                    in_path = fpath,
                    current_path = fpath,
                    info = info,
                    status = 'created',
                    flexible_residue_first = info.get('residue_first', None),
                    flexible_residue_last  = info.get('residue_last', None),
                    # H_chain = H_chain,  ##LZH Changed
                    # L_chain = L_chain,  ##LZH Changed
                    # A_chain = A_chain,  ##LZH Changed
                ))
                self.visited.add(fpath)
                
                # print(tasks[-1])
                # assert False,'debug'
        return tasks


# RelaxTask(in_path='./results/codesign_multicdrs/0000_8ds5_C_B_A_2025_12_19__19_22_40/MultipleCDRs/REF1.pdb', 
#           current_path='./results/codesign_multicdrs/0000_8ds5_C_B_A_2025_12_19__19_22_40/MultipleCDRs/REF1.pdb', 
#           info={'name': '8ds5_C_B_A-MultipleCDRs', 'tag': 'MultipleCDRs', 
#                 'cdrs': ['H_CDR1', 'H_CDR2', 'H_CDR3', 'L_CDR1', 'L_CDR2', 'L_CDR3'], 
#                 'residue_first': None, 'residue_last': None}, status='created', flexible_residue_first=None, 
#                 flexible_residue_last=None)
    # H1 = (26, 32)
    # H2 = (52, 56)
    # H3 = (95, 102)
    
    # L1 = (24, 34)
    # L2 = (50, 56)
    # L3 = (89, 97)