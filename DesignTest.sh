for i in {0..19}
do
   python design_testset.py -c ./configs/test/moe/codesign_single_H3_0.4.yml -b 32 $i
done


python diffab/tools/relax/run.py --root ./results/codesign_single_H3_0.4  --pipeline pyrosetta
python  diffab/tools/eval/run.py  --root  ./results/codesign_single_H3_0.4 --pfx rosetta


# for i in {0..19}
# do
#    python design_testset.py -c ./configs/test/moe/codesign_multicdrs_0.4.yml -b 32 $i
# done

# python diffab/tools/relax/run.py --root ./results/codesign_multicdrs_0.4  --pipeline pyrosetta
# python  diffab/tools/eval/run.py  --root  ./results/codesign_multicdrs_0.4 --pfx rosetta


