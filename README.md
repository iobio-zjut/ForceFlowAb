# ForceFlowAb

ForceFlowAb is a research codebase for antibody sequence and structure design with rectified flow. This public repository focuses on inference workflows for single-CDR and multi-CDR design, optional mixture-of-experts (MoE) routing, energy-guided sampling, and antibody-antigen docking.

![Pipeline of ForceFlowAb](assets/pipeline-of-forceflowab.jpg)

## Installation

The provided environment targets Python 3.8, PyTorch 1.12.1, and CUDA 11.3. Create it with Conda:

```bash
conda env create -f env.yaml
conda activate ForceFlowAb
```

The repository includes `data/sabdab_summary_all.tsv`, a snapshot of the SAbDab index. Download the corresponding antibody structure files separately, place them under `data/`.

## Pretrained Weights

The pretrained checkpoints are hosted on https://huggingface.co/SherrySherry123/ForceFlowAb

## Inference

Inference behavior is controlled by files under `configs/test/`. Before running inference, set `model.checkpoint` in the selected configuration to a local checkpoint. 
### Design on a test split

```bash
cd ~/ForceFlowAb
bash DesignTest.sh
```

### Design from a PDB structure

```bash
python design_pdb.py /path/to/antibody_antigen.pdb \
  --heavy H \
  --light L \
  --config ./configs/test/H3.yml \
  --out_root ./results
```

`--heavy` and `--light` specify the antibody heavy- and light-chain IDs in the input PDB. For a nanobody, provide only `--heavy`.


## Citation

If you use ForceFlowAb in academic work, please add the project citation here once the corresponding paper or preprint is available.
