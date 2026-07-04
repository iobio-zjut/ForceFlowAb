# ForceFlowAb

ForceFlowAb is a research codebase for antibody sequence and structure design with rectified flow. The repository contains training and inference pipelines for single-CDR and multi-CDR design, optional mixture-of-experts (MoE) routing, energy-guided sampling, and antibody-antigen docking workflows.


## Features

- Joint antibody sequence and backbone structure generation with rectified flow.
- Single-CDR and multi-CDR design modes.
- Configurable MoE routing with routed and shared experts.
- Two-stage training for joint sequence-structure learning and sequence-focused fine-tuning.
- Optional energy guidance during sampling.
- Design from test sets, user-provided PDB structures, or HDOCK-generated antibody-antigen poses.

## Repository layout

```text
ForceFlowAb/
|-- configs/                 # Training and inference configurations
|-- diffab/                  # Models, datasets, geometry, sampling, and evaluation code
|-- bin/                     # Optional external docking executables
|-- train.py                 # Main training entry point
|-- train_sec.py             # Second-stage training entry point
|-- train_two_stage.sh       # Two-stage training wrapper
|-- design_pdb.py            # Design from a PDB structure
|-- design_testset.py        # Design on a configured test split
|-- design_dock.py           # Docking followed by antibody design
`-- env.yaml                 # Conda environment specification
```

## Installation

The provided environment targets Python 3.8, PyTorch 1.12.1, and CUDA 11.3. Create it with Conda:

```bash
conda env create -f env.yaml
conda activate ForceFlowAb
```

The repository includes `data/sabdab_summary_all.tsv`, a snapshot of the SAbDab index used by the example configurations. Download the corresponding antibody structure files separately, place them under `data/`, and update the dataset paths in the selected YAML configuration. Large structure datasets remain excluded from version control.

## Pretrained Weights

The pretrained checkpoints are hosted on https://huggingface.co/SherrySherry123/ForceFlowAb

## Training

Run a single training stage with a YAML configuration:

```bash
python train.py configs/train/codesign_muti_rectflow_RF.yml
```

Useful options include:

```text
--logdir PATH       Directory for logs and checkpoints (default: ./logs)
--device DEVICE     Training device (default: cuda)
--num_workers N     Data-loader workers (default: 8)
--resume PATH       Resume from a checkpoint
--finetune PATH     Fine-tune from a checkpoint
--debug             Disable persistent experiment logging
```

For the two-stage workflow:

```bash
ACTIVATE_ENV=0 \
STAGE1_CONFIG=./configs/train/codesign_muti_rectflow_RF.yml \
STAGE2_CONFIG=./configs/train/codesign_muti_rectflow_finetune_RF.yml \
bash ./train_two_stage.sh
```

To resume the two-stage workflow:

```bash
RESUME_CKPT=/path/to/checkpoints/checkpoint_best.pt \
RESUME_STAGE=auto \
ACTIVATE_ENV=0 \
bash ./train_two_stage.sh
```

## Inference

Inference behavior is controlled by files under `configs/test/`. Before running inference, set `model.checkpoint` in the selected configuration to a local trained checkpoint.

### Design on a test split

```bash
python design_testset.py 0 \
  --config ./configs/test/moe/codesign_single_H3_0.4.yml \
  --out_root ./results
```

Here, `0` is the zero-based index of the structure in the configured test split.

### Design from a PDB structure

```bash
python design_pdb.py /path/to/antibody_antigen.pdb \
  --heavy H \
  --light L \
  --config ./configs/test/moe/codesign_single_H3_0.4.yml \
  --out_root ./results
```

`--heavy` and `--light` specify the antibody heavy- and light-chain IDs in the input PDB. All remaining chains are treated as antigen chains. By default, the script applies Chothia renumbering and can detect the first heavy and light chains automatically; specifying the IDs explicitly is recommended. When using `--no_renumber`, at least one antibody chain ID must be provided. For a nanobody, provide only `--heavy`.

Run `python design_pdb.py --help` or `python design_testset.py --help` for the complete set of arguments.

### Docking-guided design

`design_dock.py` can generate antibody-antigen poses with HDOCK and then run the design pipeline. The bundled `bin/hdock` and `bin/createpl` executables belong to the [HDOCK](http://hdock.phys.hust.edu.cn/) docking suite developed by Professor Sheng-You Huang's group at the School of Physics, Huazhong University of Science and Technology. Alternative executable paths can be supplied with `--hdock_bin` and `--createpl_bin`.

```bash
python design_dock.py \
  --antigen /path/to/antigen.pdb \
  --antibody /path/to/antibody.pdb \
  --heavy H \
  --light L \
  --cdrs H1 H2 H3 L1 L2 L3 \
  --epitope_sites A:991 A:992 \
  --config ./configs/test/moe/codesign_multicdrs_0.4.yml \
  --num_docks 10 \
  --out_root ./results
```

Here, `--heavy H` and `--light L` identify the antibody chains. Antigen chains come from the file passed to `--antigen`; chain IDs used in `--epitope_sites` refer to that antigen PDB (for example, residues 991 and 992 on antigen chain `A`). The default antibody chain IDs are `H` and `L`, but they should be changed when the input PDB uses different IDs.

HDOCK is third-party software and is not installed by `env.yaml`. Users should follow the official HDOCK terms and cite the original work:


## Acknowledgements

The repository structure and several core components build on the [DiffAb](https://github.com/luost26/diffab) antibody-design codebase. Its flow-matching approach to antibody CDR sequence-structure co-design also draws on [FlowDesign](https://github.com/nohandsomewujun/FlowDesign).

The docking workflow uses HDOCK from Professor Sheng-You Huang's group at the School of Physics, Huazhong University of Science and Technology. We thank the HDOCK authors for making their docking tools available to the academic community.

> Luo, S. *et al.* Antigen-Specific Antibody Design and Optimization with Diffusion-Based Generative Models for Protein Structures. *Advances in Neural Information Processing Systems* **35** (2022). [NeurIPS paper](https://papers.neurips.cc/paper_files/paper/2022/hash/3fa7d76a0dc1179f1e98d1bc62403756-Abstract-Conference.html)

> Wu, J. *et al.* FlowDesign: Improved design of antibody CDRs through flow matching and better prior distributions. *Cell Systems* **16**, 101270 (2025). [https://doi.org/10.1016/j.cels.2025.101270](https://doi.org/10.1016/j.cels.2025.101270)

> Yan, Y., Zhang, D., Zhou, P., Li, B. & Huang, S.-Y. HDOCK: a web server for protein-protein and protein-DNA/RNA docking based on a hybrid strategy. *Nucleic Acids Research* **45**, W365-W373 (2017). [https://doi.org/10.1093/nar/gkx407](https://doi.org/10.1093/nar/gkx407)

## License

The original ForceFlowAb source code is released under the [MIT License](LICENSE). Third-party code, bundled HDOCK executables, and SAbDab-derived metadata are not relicensed by this repository and remain subject to their respective licenses and terms of use.

## Citation

If you use ForceFlowAb in academic work, please add the project citation here once the corresponding paper or preprint is available.
