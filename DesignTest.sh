for i in {0..19}
do
   python design_testset.py -c ./configs/test/H3.yml -b 32 $i
done


python diffab/tools/relax/run.py --root ./results/H3  --pipeline pyrosetta
python  diffab/tools/eval/run.py  --root  ./results/H3 --pfx rosetta


# for i in {0..19}
# do
#    python design_testset.py -c ./configs/test/multicdrs.yml -b 32 $i
# done

# python diffab/tools/relax/run.py --root ./results/multicdrs  --pipeline pyrosetta
# python  diffab/tools/eval/run.py  --root  ./results/multicdrs --pfx rosetta


