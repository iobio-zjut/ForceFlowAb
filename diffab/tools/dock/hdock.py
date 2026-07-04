import os
import csv
import shutil
import tempfile
import subprocess
import dataclasses as dc
import re
from typing import List, Optional, Sequence
from Bio import PDB
from Bio.PDB import Model as PDBModel

from diffab.tools.renumber import renumber as renumber_chothia
from .base import DockingEngine


CDR_SITES = {
    'H1': (26, 32, 'heavy'),
    'H2': (52, 56, 'heavy'),
    'H3': (95, 102, 'heavy'),
    'L1': (24, 34, 'light'),
    'L2': (50, 56, 'light'),
    'L3': (89, 97, 'light'),
}

FLOAT_LINE_RE = re.compile(
    r"^\s*[-+]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?(?:\s+[-+]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?){8,}\s*$"
)


def fix_docked_pdb(pdb_path):
    fixed = []
    with open(pdb_path, 'r') as f:
        for ln in f.readlines():
            if (ln.startswith('ATOM') or ln.startswith('HETATM')) and len(ln) == 56:
                fixed.append( ln[:-1] + ' 1.00  0.00              \n' )
            else:
                fixed.append(ln)
    with open(pdb_path, 'w') as f:
        f.write(''.join(fixed))


class HDock(DockingEngine):

    def __init__(
        self, 
        hdock_bin='./bin/hdock',
        createpl_bin='./bin/createpl',
    ):
        super().__init__()
        self.hdock_bin = os.path.realpath(hdock_bin)
        self.createpl_bin = os.path.realpath(createpl_bin)
        self.tmpdir = tempfile.TemporaryDirectory()

        self._has_receptor = False
        self._has_ligand = False

        self._receptor_chains = []
        self._ligand_chains = []
        self._last_hdock_out_path = None
        self._last_score_rows = []

    def __enter__(self):
        return self

    def __exit__(self, typ, value, traceback):
        self.tmpdir.cleanup()

    def set_receptor(self, pdb_path):
        shutil.copyfile(pdb_path, os.path.join(self.tmpdir.name, 'receptor.pdb'))
        self._has_receptor = True

    def set_ligand(self, pdb_path):
        shutil.copyfile(pdb_path, os.path.join(self.tmpdir.name, 'ligand.pdb'))
        self._has_ligand = True

    def _dump_complex_pdb(self):
        parser = PDB.PDBParser(QUIET=True)
        model_receptor = parser.get_structure(None, os.path.join(self.tmpdir.name, 'receptor.pdb'))[0]
        docked_pdb_path = os.path.join(self.tmpdir.name, 'ligand_docked.pdb')
        fix_docked_pdb(docked_pdb_path)
        structure_ligdocked = parser.get_structure(None, docked_pdb_path)

        pdb_io = PDB.PDBIO()
        paths = []
        for i, model_ligdocked in enumerate(structure_ligdocked):
            model_complex = PDBModel.Model(0)
            for chain in model_receptor:
                model_complex.add(chain.copy())
            for chain in model_ligdocked:
                model_complex.add(chain.copy())
            pdb_io.set_structure(model_complex)
            save_path = os.path.join(self.tmpdir.name, f"complex_{i}.pdb")
            pdb_io.save(save_path)
            paths.append(save_path)
        return paths

    def _parse_hdock_scores(self, hdock_out_path):
        rows = []
        with open(hdock_out_path, 'r') as handle:
            for line in handle:
                if FLOAT_LINE_RE.match(line) is None:
                    continue
                fields = line.split()
                if len(fields) < 9:
                    continue
                rank = len(rows) + 1
                rows.append({
                    'rank': rank,
                    'score': float(fields[6]),
                    'raw_line': line.rstrip('\n'),
                })
        return rows

    def export_last_outputs(self, output_dir, pdb_paths, start_rank=1):
        if self._last_hdock_out_path is None or not os.path.exists(self._last_hdock_out_path):
            raise FileNotFoundError('Hdock.out is not available for export.')

        os.makedirs(output_dir, exist_ok=True)
        hdock_out_dest = os.path.join(output_dir, 'Hdock.out')
        shutil.copyfile(self._last_hdock_out_path, hdock_out_dest)

        scores_csv = os.path.join(output_dir, 'dock_scores.csv')
        with open(scores_csv, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['rank', 'score', 'pdb_path', 'raw_line'])
            writer.writeheader()
            for row_idx, pdb_path in enumerate(pdb_paths, start=start_rank):
                row = {'rank': row_idx, 'score': '', 'pdb_path': pdb_path, 'raw_line': ''}
                if row_idx <= len(self._last_score_rows):
                    row.update(self._last_score_rows[row_idx - 1])
                writer.writerow(row)

    def dock(self, nmax=None):
        if not (self._has_receptor and self._has_ligand):
            raise ValueError('Missing receptor or ligand.')
        subprocess.run(
            [self.hdock_bin, "receptor.pdb", "ligand.pdb"],
            cwd=self.tmpdir.name, check=True
        )
        cmd_pl = [self.createpl_bin, "Hdock.out", "ligand_docked.pdb"]
        if nmax is not None:
            cmd_pl += ["-nmax", str(int(nmax))]
        subprocess.run(cmd_pl, cwd=self.tmpdir.name, check=True)
        self._last_hdock_out_path = os.path.join(self.tmpdir.name, 'Hdock.out')
        self._last_score_rows = self._parse_hdock_scores(self._last_hdock_out_path)
        return self._dump_complex_pdb()


@dc.dataclass
class DockSite:
    chain: str
    resseq: int


class HDockAntibody(HDock):

    def __init__(self, *args, cdrs: Optional[Sequence[str]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._heavy_chain_id = None
        self._light_chain_id = None
        self._epitope_sites: Optional[List[DockSite]] = None
        self._cdrs = list(cdrs) if cdrs is not None else ['H3']

    def set_ligand(self, pdb_path):
        raise NotImplementedError('Please use set_antibody')
    
    def set_receptor(self, pdb_path):
        raise NotImplementedError('Please use set_antigen')

    def set_antigen(self, pdb_path, epitope_sites: Optional[List[DockSite]]=None):
        super().set_receptor(pdb_path)
        self._epitope_sites = epitope_sites

    def set_antibody(self, pdb_path):
        heavy_chains, light_chains = renumber_chothia(pdb_path, os.path.join(self.tmpdir.name, 'ligand.pdb'))
        self._has_ligand = True
        self._heavy_chain_id = heavy_chains[0] if len(heavy_chains) > 0 else None
        self._light_chain_id = light_chains[0] if len(light_chains) > 0 else None

    def _cdr_site_to_line(self, cdr_name: str) -> str:
        if cdr_name not in CDR_SITES:
            raise ValueError(f'Unknown CDR region: {cdr_name}')
        start, end, chain_kind = CDR_SITES[cdr_name]
        chain_id = self._heavy_chain_id if chain_kind == 'heavy' else self._light_chain_id
        if chain_id is None:
            raise ValueError(f'CDR {cdr_name} was selected, but no corresponding {chain_kind} chain was found.')
        return f'{start}-{end}:{chain_id}\n'

    def _prepare_lsite(self):
        selected_cdrs = []
        seen = set()
        for cdr_name in self._cdrs:
            if cdr_name not in seen:
                selected_cdrs.append(cdr_name)
                seen.add(cdr_name)

        lsite_content = ''.join(self._cdr_site_to_line(cdr_name) for cdr_name in selected_cdrs)
        with open(os.path.join(self.tmpdir.name, 'lsite.txt'), 'w') as f:
            f.write(lsite_content)
        print(f"[INFO] lsite content: {lsite_content}")

    def _prepare_rsite(self):
        rsite_content = ""
        for site in self._epitope_sites:
            rsite_content += f"{site.resseq}:{site.chain}\n"
        with open(os.path.join(self.tmpdir.name, 'rsite.txt'), 'w') as f:
            f.write(rsite_content)
        print(f"[INFO] rsite content: {rsite_content}")

    def dock(self, nmax=None):
        if not (self._has_receptor and self._has_ligand):
            raise ValueError('Missing receptor or ligand.')
        self._prepare_lsite()

        cmd_hdock = [self.hdock_bin, "receptor.pdb", "ligand.pdb", "-lsite", "lsite.txt"]
        if self._epitope_sites is not None:
            self._prepare_rsite()
            cmd_hdock += ["-rsite", "rsite.txt"]
        subprocess.run(
            cmd_hdock,
            cwd=self.tmpdir.name, check=True
        )

        cmd_pl = [self.createpl_bin, "Hdock.out", "ligand_docked.pdb", "-lsite", "lsite.txt"]
        if self._epitope_sites is not None:
            self._prepare_rsite()
            cmd_pl += ["-rsite", "rsite.txt"]
        if nmax is not None:
            cmd_pl += ["-nmax", str(int(nmax))]
        subprocess.run(
            cmd_pl, 
            cwd=self.tmpdir.name, check=True
        )
        self._last_hdock_out_path = os.path.join(self.tmpdir.name, 'Hdock.out')
        self._last_score_rows = self._parse_hdock_scores(self._last_hdock_out_path)
        return self._dump_complex_pdb()


if __name__ == '__main__':
    with HDockAntibody('hdock', 'createpl') as dock:
        dock.set_antigen('./data/dock/receptor.pdb', [DockSite('A', 991)])
        dock.set_antibody('./data/example_dock/3qhf_fv.pdb')
        print(dock.dock())
