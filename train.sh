encoder=$ENCODER
epochs=5
bs=16
loss=hardtriplet
setting=unique
eval_steps=$EVALSTEPS

source activate py38

# for lang in en da de es fi fr it nl pl pt sv; do
for lang in en; do
	outdir=./output/${encoder}_${lang}_${setting}_${loss}_${epochs}_${bs}
	python -m sentence_transformers.train --loss $loss --data_path ./data/analogy_dists_splits --train_data analogy_${setting}_${lang}_dists.csv.train --dev_data analogy_${setting}_en_dists.csv.valid --bs $bs --encoder $encoder --epochs $epochs --evaluation_steps $eval_steps --out $outdir
done
