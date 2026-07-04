#!/bin/bash

. activate diffab
cd /xsdata/lzhlzh/26_36/FlowAB/FlowDesign

# python design_dock.py \
#     --antigen ./data/dock/rsv_site3.pdb \
#     --antibody ./data/dock/hu-4D5-8_Fv.pdb \
#     --cdrs H1 H2 H3 L1 L2 L3 \
#     --epitope_sites T:305 T:456 \
#     --config ./data/dock/hu-4D5-8_Fv.yml \
#     --heavy H \
#     --light L \
#     --num_docks 4 \
#     -b 1


# python diffab/tools/relax/run.py --root ./results/hu-4D5-8_Fv_rsv_site3 --pipeline pyrosetta
# python  diffab/tools/eval/run.py  --root  ./results/hu-4D5-8_Fv_rsv_site3 --pfx rosetta

# python design_dock.py \
#     --antigen ./data/dock/flu_HA.pdb \
#     --antibody ./data/dock/h-NbBCII10.pdb \
#     --cdrs H1 H2 H3 \
#     --epitope_sites B:146 B:170 B:177 \
#     --config ./data/dock/h-NbBCII10.yml \
#     --heavy H \
#     --num_docks 1000 \
#     -b 1




# python diffab/tools/relax/run.py --root ./results/h-NbBCII10_flu_HA --pipeline pyrosetta
python  diffab/tools/eval/run.py  --root  ./results/h-NbBCII10_flu_HA --pfx rosetta