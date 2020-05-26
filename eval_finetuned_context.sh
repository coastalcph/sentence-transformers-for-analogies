seed=$SEED
encoder=$ENCODER
bs=16
epochs=5
loss=hardtriplet
setting=$SETTING
part=test
context=$CONTEXT
name=$NAME

for lang in en da de es fi fr it nl pl pt sv; do
# for lang in en; do
	indir=./output/${name}_context_${context}_${encoder}_whole_${setting}_${loss}_${epochs}_${bs}_${seed}
	#indir=./output/${name}_context_${context}_${encoder}_${lang}_${setting}_${loss}_${epochs}_${bs}_${seed}
	outdir=./output/${name}_context_${context}_${encoder}_${lang}_${setting}_${loss}_${epochs}_${bs}_${seed}_${part}_predictions_corr
	python -m sentence_transformers.predict_from_finetuned --data_path ./data/analogy_contexts_splits --test_data analogy_${setting}_${lang}_contexts.csv.${part} --bs $bs --model $indir --out $outdir --distance_file ./data/analogy_contexts_fix/analogy_${setting}_${lang}_contexts.csv --context $context
done
