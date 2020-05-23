encoder=$ENCODER
bs=16
setting=$SETTING
part=test

for lang in en da de es fi fr it nl pl pt sv; do
# for lang in en; do
	outdir=./output/context_pretrained_${encoder}_${lang}_${setting}_${part}_corr
	python -m sentence_transformers.predict --data_path ./data/analogy_contexts_splits --test_data analogy_${setting}_${lang}_contexts.csv.${part} --bs $bs --encoder $encoder --out $outdir --distance_file ./data/analogy_contexts_fix/analogy_${setting}_${lang}_contexts.csv
done

