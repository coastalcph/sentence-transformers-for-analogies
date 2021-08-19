# Sentence Transformers for Analogies

This is the code repository for the paper [Analogy Training Multilingual Encoders](https://ojs.aaai.org/index.php/AAAI/article/view/17524/17331)

It is a fork from the original [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) repository.

It contains three training scripts;
1. `train.sh`: Trains a sentence transformer using only the entities of a given analogy.
2. `train_alias.sh`: Trains a sentence transformer using the entities along with their respective aliases.
3. `train_context.sh`: Trains a sentence transformer using the entities along with their respective contexts (descriptions).

It contains three scripts evaluating pre-trained architectures;
1. `eval_pretrained.sh`: Evaluate a pre-trained sentence transformer using only the entities of a given analogy.
2. `eval_pretrained_alias.sh`: Evaluate a pre-trained sentence transformer using the entities along with their respective aliases.
3. `eval_pretrained_context.sh`: Evaluate a pre-trained sentence transformer using the entities along with their respective contexts (descriptions).

It contains three scripts evaluating fine-tuned architectures;
1. `eval_finetuned.sh`: Evaluate a fine-tuned sentence transformer using only the entities of a given analogy.
2. `eval_finetuned_alias.sh`: Evaluate a fine-tuned sentence transformer using the entities along with their respective aliases.
3. `eval_finetuned_context.sh`: Evaluate a fine-tuned sentence transformer using the entities along with their respective contexts (descriptions).
