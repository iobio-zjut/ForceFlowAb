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

### Optional: HDOCK

HDOCK is only required for workflows that dock an antibody template to an antigen. Download `hdock` and `createpl` from the [official HDOCK site](http://huanglab.phys.hust.edu.cn/software/hdocklite/) and place them in `bin/`:

```text
bin/
├── hdock
└── createpl
```


## Train

```bash
ACTIVATE_ENV=0 \
STAGE1_CONFIG=./configs/train/codesign_muti_rectflow_RF.yml \
STAGE2_CONFIG=./configs/train/codesign_muti_rectflow_finetune_RF.yml \
bash ./train_two_stage.sh
```

## Trained Weights

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

If you use ForceFlowAb in academic work, please cite the archived software release listed in the repository's `CITATION.cff` file. The version-specific Zenodo DOI will be added after the first archival release.

## License and third-party software

ForceFlowAb-specific contributions are provided under the MIT License. Portions of this repository are derived from [DiffAb](https://github.com/luost26/diffab) and remain subject to the Apache License 2.0 and its attribution requirements. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and [`licenses/DiffAb-Apache-2.0.txt`](licenses/DiffAb-Apache-2.0.txt) for details.
