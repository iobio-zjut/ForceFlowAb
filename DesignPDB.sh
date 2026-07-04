

# python -m diffab.tools.runner.design_for_pdb \
#     --pdb_path /path/to/complex.pdb \
#     --heavy H \
#     --light L \
#     -c ./configs/test/moe/codesign_single_H3_0.4.yml \
#     -o ./results \
#     -b 32

# #   如果是 nanobody，没有轻链，可以只给重链，例如：

#   python -m diffab.tools.runner.design_for_pdb \
#     --pdb_path /path/to/nanobody_complex.pdb \
#     --heavy H \
#     --light '' \
#     -c ./configs/test/moe/codesign_single_H3_0.4.yml \
#     -o ./results \
#     -b 32




# SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# cd "$SCRIPT_DIR" || exit 1

# INPUT_DIR="$SCRIPT_DIR/results/rfdiffusion_backbones"
# RESULT_DIR="$SCRIPT_DIR/results/RfabBockbone"

# is_finished() {
#     local fname=$1
#     local finished_pdb

#     for finished_pdb in "$RESULT_DIR"/"$fname"_*/MultipleCDRs/0000.pdb
#     do
#         [[ -s "$finished_pdb" ]] && return 0
#     done

#     return 1
# }

# for pdb in "$INPUT_DIR"/*.pdb
# do
#     fname=$(basename "$pdb")

#     [[ "$fname" == *chothia* ]] && continue

#     if is_finished "$fname"; then
#         echo "Skip finished: $fname"
#         continue
#     fi

#     python -m diffab.tools.runner.design_for_pdb \
#       "$pdb" \
#       --heavy H \
#       --light L \
#       -c ./configs/test/RfabBockbone.yml \
#       -o ./results \
#       -b 1
# done

# python diffab/tools/relax/run.py --root ./results/RfabBockbone --pipeline pyrosetta
# python  diffab/tools/eval/run.py  --root  ./results/RfabBockbone --pfx rosetta

python -m diffab.tools.runner.design_for_pdb \
  "/xsdata/lzhlzh/26_36/FlowAB/FlowDesign/results/526/FF438.pdb" \
  --heavy H \
  --light L \
  -c ./configs/test/RfabBockbone.yml \
  -o ./results \
  -b 1
python diffab/tools/relax/run.py --root ./results/RfabBockbone --pipeline pyrosetta
python  diffab/tools/eval/run.py  --root  ./results/RfabBockbone --pfx rosetta