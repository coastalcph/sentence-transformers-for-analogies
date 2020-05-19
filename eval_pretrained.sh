encoder=$ENCODER
bs=16
setting=$SETTING
part=test

for lang in en da de es fi fr it nl pl pt sv; do
# for lang in en; do
	outdir=./output/pretrained_${encoder}_${lang}_${setting}_${part}_corr
	python -m sentence_transformers.predict --data_path ./data/analogy_dists_splits --test_data analogy_${setting}_${lang}_dists.csv.${part} --bs $bs --encoder $encoder --out $outdir --distance_file ./data/analogy_dists/analogy_${setting}_${lang}_dists.csv
done

