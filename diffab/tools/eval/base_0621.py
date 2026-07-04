import os
import re
import json
import shelve
import logging
from Bio import PDB
from typing import Optional, Tuple, List
from dataclasses import dataclass, field


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
class EvalTask:
    in_path: str
    ref_path: str
    info: dict
    structure: str
    name: str
    method: str
    cdr: str
    ab_chains: List

    residue_first: Optional[Tuple] = None
    residue_last: Optional[Tuple] = None
    
    scores: dict = field(default_factory=dict)

    def get_gen_biopython_model(self):
        parser = PDB.PDBParser(QUIET=True)
        return parser.get_structure(self.in_path, self.in_path)[0]

    def get_ref_biopython_model(self):
        parser = PDB.PDBParser(QUIET=True)
        return parser.get_structure(self.ref_path, self.ref_path)[0]

    def save_to_db(self, db: shelve.Shelf):
        db[self.in_path] = self

    def to_report_dict(self):
        return {
            'method': self.method,
            'structure': self.structure,
            'cdr': self.cdr,
            'filename': os.path.basename(self.in_path),
            **self.scores
        }


class TaskScanner:

    def __init__(self, root, postfix=None, db: Optional[shelve.Shelf]=None):
        super().__init__()
        self.root = root
        self.postfix = postfix
        self.visited = set()
        self.db = db
        if db is not None:
            for k in db.keys():
                self.visited.add(k)

    def _get_metadata(self, fpath):
        json_path = os.path.join(
            os.path.dirname(os.path.dirname(fpath)), 
            'metadata.json'
        )
        tag_name = os.path.basename(os.path.dirname(fpath))
        method_name = os.path.basename(
            os.path.dirname(os.path.dirname(os.path.dirname(fpath)))
        )
        # print(method_name, tag_name)
        # assert  False,'debug'
        try:
            info = None
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            for item in metadata['items']:
                if item['tag'] == tag_name:
                    info = item
            if info is not None:
                if tag_name == 'MultipleCDRs':
                    info.update(self._build_multicdr_metadata(fpath, info))
                else:
                    info['antibody_chains'] = [info['residue_first'][0]]
                info['structure'] = metadata['identifier']
                info['method'] = method_name
            return info
        except (json.JSONDecodeError, FileNotFoundError) as e:
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

    def _infer_multicdr_chain_ids(self, fpath, info):
        chain_resseqs = self._parse_chain_resseqs_from_pdb(fpath)
        selected_cdrs = self._normalize_cdrs(info.get('cdrs', []))
        heavy_markers = {start for name, chain_kind, start, _ in MULTI_CDR_REGION_DEFS if chain_kind == 'heavy' and name in selected_cdrs}
        light_markers = {start for name, chain_kind, start, _ in MULTI_CDR_REGION_DEFS if chain_kind == 'light' and name in selected_cdrs}

        heavy_chain = None
        light_chain = None

        if heavy_markers:
            for chain_id, resseqs in chain_resseqs.items():
                if heavy_markers.issubset(resseqs):
                    heavy_chain = chain_id
                    break
        if light_markers:
            for chain_id, resseqs in chain_resseqs.items():
                if chain_id == heavy_chain:
                    continue
                if light_markers.issubset(resseqs):
                    light_chain = chain_id
                    break

        parsed_from_name = []
        name = info.get('name', '')
        if name:
            parsed_from_name = name.split('_')
        if heavy_markers and heavy_chain is None and len(parsed_from_name) > 1 and len(parsed_from_name[1]) == 1:
            heavy_chain = parsed_from_name[1]
        if light_markers and light_chain is None and len(parsed_from_name) > 2 and len(parsed_from_name[2]) == 1:
            light_chain = parsed_from_name[2]

        antigen_chain = None
        for chain_id in chain_resseqs:
            if chain_id not in {heavy_chain, light_chain}:
                antigen_chain = chain_id
                break
        if antigen_chain is None and len(parsed_from_name) > 3:
            parsed_antigen = parsed_from_name[3].split('-')[0]
            if len(parsed_antigen) == 1:
                antigen_chain = parsed_antigen

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

        antibody_chains = []
        if heavy_chain is not None:
            antibody_chains.append(heavy_chain)
        if light_chain is not None:
            antibody_chains.append(light_chain)

        return {
            'residue_first': residue_first,
            'residue_last': residue_last,
            'antibody_chains': antibody_chains,
        }

    def scan(self) -> List[EvalTask]: 
        tasks = []
        if self.postfix is None or not self.postfix:
            input_fname_pattern = '^\d+\.pdb$'
            ref_fname = 'REF1.pdb'
        else:
            # input_fname_pattern = f'^\d+\_{self.postfix}\.pdb$'
            input_fname_pattern = rf'^(\d+_{self.postfix}\.pdb$|P.*{self.postfix}.*\.pdb$)'
            ref_fname = f'REF1_{self.postfix}.pdb'
        for parent, _, files in os.walk(self.root):
            for fname in files:
                fpath = os.path.join(parent, fname)
                if not re.match(input_fname_pattern, fname):
                    continue
                if os.path.getsize(fpath) == 0:
                    continue
                if fpath in self.visited:
                    continue

                # Path to the reference structure
                ref_path = os.path.join(parent, ref_fname)
                if not os.path.exists(ref_path):
                    continue

                # CDR information
                info = self._get_metadata(fpath)
                if info is None:
                    continue
                tasks.append(EvalTask(
                    in_path = fpath,
                    ref_path = ref_path,
                    info = info,
                    structure = info['structure'],
                    name = info['name'],
                    method = info['method'],
                    cdr = info['tag'],
                    ab_chains = info['antibody_chains'],
                    residue_first = info.get('residue_first', None),
                    residue_last  = info.get('residue_last', None),
                ))
                self.visited.add(fpath)
                # print(tasks[-1])
                # assert False,'debug'
        return tasks


# EvalTask(in_path='results/codesign_multicdrs/0000_8ds5_C_B_A_2025_12_19__19_22_40/MultipleCDRs/0000_rosetta.pdb', 
# ref_path='results/codesign_multicdrs/0000_8ds5_C_B_A_2025_12_19__19_22_40/MultipleCDRs/REF1_rosetta.pdb', 
# info={'name': '8ds5_C_B_A-MultipleCDRs', 'tag': 'MultipleCDRs', 'cdrs': ['H_CDR1', 'H_CDR2', 'H_CDR3', 'L_CDR1', 'L_CDR2', 'L_CDR3'], 
# 'residue_first': [95, 52, 26, 89, 50, 24, 'C', 'B', 'A'], 'residue_last': [102, 56, 32, 97, 56, 34, 'C', 'B', 'A'], 
# 'antibody_chains': ['B', 'C'], 'structure': '8ds5_C_B_A', 'method': 'codesign_multicdrs'}, structure='8ds5_C_B_A', name='8ds5_C_B_A-MultipleCDRs',
#  method='codesign_multicdrs', cdr='MultipleCDRs', ab_chains=['B', 'C'], residue_first=[95, 52, 26, 89, 50, 24, 'C', 'B', 'A'],
#  residue_last=[102, 56, 32, 97, 56, 34, 'C', 'B', 'A'], scores={})
