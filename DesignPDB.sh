

# python -m diffab.tools.runner.design_for_pdb \
#     --pdb_path /path/to/complex.pdb \
#     --heavy H \
#     --light L \
#     -c ./configs/test/moe/codesign_single_H3_0.4.yml \
#     -o ./results \
#     -b 32


# #  nanobody
#   python -m diffab.tools.runner.design_for_pdb \
#     --pdb_path /path/to/nanobody_complex.pdb \
#     --heavy H \
#     --light '' \
#     -c ./configs/test/moe/codesign_single_H3_0.4.yml \
#     -o ./results \
#     -b 32




python -m diffab.tools.runner.design_for_pdb \
  /xsdata/lzhlzh/26_36/FlowAB/FlowDesign/testwk/7che.pdb \
  --heavy H \
  --light L \
  -c ./configs/test/multicdrs.yml \
  -o ./testwk \
  -b 8

python -m diffab.tools.runner.design_for_pdb \
  /xsdata/lzhlzh/26_36/FlowAB/FlowDesign/testwk/7che.pdb \
  --heavy H \
  --light L \
  -c ./configs/test/H3.yml \
  -o ./testwk \
  -b 8

python diffab/tools/relax/run.py --root ./testwk/H3 --pipeline pyrosetta
python  diffab/tools/eval/run.py  --root  ./testwk/H3 --pfx rosetta

python diffab/tools/relax/run.py --root ./testwk/multicdrs --pipeline pyrosetta
python  diffab/tools/eval/run.py  --root  ./testwk/multicdrs --pfx rosetta