seed=$SEED
encoder=$ENCODER
epochs=$EPOCHS
bs=16
loss=hardtriplet
setting=$SETTING
eval_steps=$EVALSTEPS
context=$CONTEXT
name=$NAME

# for lang in en da de es fi fr it nl pl pt sv; do
# for lang in en; do
outdir=./output/${name}_context_${context}_${encoder}_whole_${setting}_${loss}_${epochs}_${bs}_${seed}
python -m sentence_transformers.train --loss $loss --data_path ./data/analogy_contexts_splits --train_data all_languages_${setting}.csv.train --dev_data analogy_${setting}_en_contexts.csv.valid --bs $bs --encoder $encoder --epochs $epochs --evaluation_steps $eval_steps --out $outdir --seed $seed --context $context
# done
