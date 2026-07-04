import os
import re
import json
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass


MULTI_CDR_REGION_DEFS = [
    ('H3', 'heavy', 95, 102),
    ('H2', 'heavy', 52, 56),
    ('H1', 'heavy', 26, 32),
    ('L3', 'light', 89, 97),
    ('L2', 'light', 50, 56),
    ('L1', 'light', 24, 34),
]
CDR_NAME_MAP = {
    'H1': 'H1',
    'H2': 'H2',
    'H3': 'H3',
    'L1': 'L1',
    'L2': 'L2',
    'L3': 'L3',
    'H_CDR1': 'H1',
    'H_CDR2': 'H2',
    'H_CDR3': 'H3',
    'L_CDR1': 'L1',
    'L_CDR2': 'L2',
    'L_CDR3': 'L3',
}


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

    def _parse_chain_resseqs_from_pdb(self, fpath):
        chain_resseqs = {}
        with open(fpath, 'r') as handle:
            for line in handle:
                if not line.startswith(('ATOM  ', 'HETATM')):
                    continue
                if len(line) < 26:
                    continue
                chain_id = line[21].strip()
                if not chain_id:
                    continue
                resseq_str = line[22:26].strip()
                if not resseq_str:
                    continue
                try:
                    resseq = int(resseq_str)
                except ValueError:
                    continue
                chain_resseqs.setdefault(chain_id, set()).add(resseq)
        return chain_resseqs

    def _normalize_cdrs(self, cdrs):
        normalized = []
        for cdr in cdrs or []:
            short_name = CDR_NAME_MAP.get(cdr)
            if short_name is not None and short_name not in normalized:
                normalized.append(short_name)
        return normalized

    def _parse_chain_ids_from_name(self, info):
        name = info.get('name', '')
        parts = name.split('_') if name else []
        heavy_chain = parts[1] if len(parts) > 1 else None
        light_chain = parts[2] if len(parts) > 2 else None
        antigen_chain = parts[3].split('-')[0] if len(parts) > 3 else None
        return heavy_chain, light_chain, antigen_chain

    def _infer_multicdr_chain_ids(self, fpath, info):
        name = info.get('name', '')
        chain_resseqs = self._parse_chain_resseqs_from_pdb(fpath)
        selected_cdrs = self._normalize_cdrs(info.get('cdrs', []))
        heavy_markers = {start for name, chain_kind, start, _ in MULTI_CDR_REGION_DEFS if chain_kind == 'heavy' and name in selected_cdrs}
        light_markers = {start for name, chain_kind, start, _ in MULTI_CDR_REGION_DEFS if chain_kind == 'light' and name in selected_cdrs}

        expected_heavy, expected_light, antigen_chain = self._parse_chain_ids_from_name(info)
        heavy_chain = expected_heavy if heavy_markers else None
        light_chain = expected_light if light_markers else None

        if heavy_markers and heavy_chain is None:
            for chain_id, resseqs in chain_resseqs.items():
                if heavy_markers.issubset(resseqs):
                    heavy_chain = chain_id
                    break
        if light_markers and light_chain is None:
            for chain_id, resseqs in chain_resseqs.items():
                if chain_id == heavy_chain:
                    continue
                if light_markers.issubset(resseqs):
                    light_chain = chain_id
                    break

        if antigen_chain is None:
            for chain_id in chain_resseqs:
                if chain_id not in {heavy_chain, light_chain}:
                    antigen_chain = chain_id
                    break

        if heavy_markers and heavy_chain is None:
            raise ValueError(
                f'Failed to infer MultipleCDRs heavy chain from {fpath}. '
                f'Observed chains: {sorted(chain_resseqs.keys())}, name={name!r}'
            )
        if light_markers and light_chain is None:
            raise ValueError(
                f'Failed to infer MultipleCDRs light chain from {fpath}. '
                f'Observed chains: {sorted(chain_resseqs.keys())}, name={name!r}'
            )

        return heavy_chain, light_chain, antigen_chain or ''

    def _build_multicdr_metadata(self, fpath, info):
        selected_cdrs = self._normalize_cdrs(info.get('cdrs', []))
        heavy_chain, light_chain, antigen_chain = self._infer_multicdr_chain_ids(fpath, info)

        residue_first = []
        residue_last = []
        for name, chain_kind, start, end in MULTI_CDR_REGION_DEFS:
            if name not in selected_cdrs:
                continue
            residue_first.append(start)
            residue_last.append(end)

        if any(name.startswith('H') for name in selected_cdrs):
            residue_first.append(heavy_chain)
            residue_last.append(heavy_chain)
        if any(name.startswith('L') for name in selected_cdrs):
            residue_first.append(light_chain)
            residue_last.append(light_chain)
        residue_first.append(antigen_chain)
        residue_last.append(antigen_chain)

        return {
            'residue_first': residue_first,
            'residue_last': residue_last,
        }

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
                    
                if info.get('tag') == 'MultipleCDRs':
                    info.update(self._build_multicdr_metadata(fpath, info))
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
