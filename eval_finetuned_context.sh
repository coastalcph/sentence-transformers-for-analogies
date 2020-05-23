seed=$SEED
encoder=$ENCODER
bs=16
epochs=5
loss=hardtriplet
setting=$SETTING
part=test

for lang in en da de es fi fr it nl pl pt sv; do
# for lang in en; do
	indir=./output/context_${encoder}_${lang}_${setting}_${loss}_${epochs}_${bs}_${seed}
	outdir=./output/context_${encoder}_${lang}_${setting}_${loss}_${epochs}_${bs}_${seed}_${part}_predictions_corr
	python -m sentence_transformers.predict_from_finetuned --data_path ./data/analogy_contexts_splits --test_data analogy_${setting}_${lang}_contexts.csv.${part} --bs $bs --model $indir --out $outdir --distance_file ./data/analogy_contexts_fix/analogy_${setting}_${lang}_contexts.csv
done
