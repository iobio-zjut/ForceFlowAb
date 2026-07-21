# ForceFlowAb

ForceFlowAb is a rectified-flow framework for antibody sequence and structure design that integrates mixture-of-experts (MoE) modeling and energy-guided sampling.

## Pipeline of ForceFlowAb
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


### Antibody-Antigen Complex

```bash
cd ~/ForceFlowAb
python design_pdb.py /path/to/antibody_antigen.pdb \
  --heavy H \
  --light L \
  --config ./configs/test/H3.yml \
  --out_root ./results
```

`--heavy` and `--light` specify the antibody heavy- and light-chain IDs in the input PDB. For a nanobody, provide only `--heavy`.


## Citation

If you use ForceFlowAb in academic work, please add the project citation here once the corresponding paper or preprint is available.
