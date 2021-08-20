# Analogy Training Multilingual Encoders

This is the code repository for the AAAI paper [Analogy Training Multilingual Encoders](https://ojs.aaai.org/index.php/AAAI/article/view/17524/17331) (Garneau et al. 2021). It is a fork from the original [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) repository.

The goal of our work is to improve multilingual encoders, based on the observation that mutlilingual encoders are globally inconsistent, i.e. the extent to which they reflect semantic relations varies with scale. For example, in the figure below, the analogy *woman is to man as catwoman is to dude* is more diffcult to predict than *woman is to man as queen is to king*, and predictions are more difficult the further the elements of the analogy are apart from each other. 
<p align="center">
  <img src="https://github.com/coastalcph/sentence-transformers-for-analogies/blob/master/inconsistency.png" />
</p>

We hypothesize that improved global consistency improves isomorphism between the language-specific subspaces and helps cross-lingual transfer. To improve global consistency, we train a pre-trained multilingual encoder on an analogy prediction task in multiple languages. To this end, we collect **WiQueen**, a new large-scale analogy dataset in 11 languages (Danish, Dutch, English, Finnish, French, German, Italian, Polish, Portuguese, Spanish, and Swedish) based on Wikidata data.
                        
In this repository, we release 
- the [analogy dataset](https://bit.ly/3aaKTzF) used to train the multilingual encoder
- the [improved multilingual encoder]() (which is a version of the mBert encoder trained on our analogy dataset)
- scripts for training and evaluating models

## Scripts

The repository contains the training and evaluation scripts that we used for the experiments reported in the paper.

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

## Citing and Contact 
If you have questions or comments, please contact the corresponding author at `nicolas.garneau@ift.ulaval.ca` (Nicolas Garneau) or `mrkhartmann4@gmail.com` (Mareike Hartmann).

The paper can be cited as 
```
@inproceedings{garneau2021analogy,
  title={Analogy Training Multilingual Encoders},
  author={Garneau, Nicolas and Hartmann, Mareike and Sandholm, Anders and Ruder, Sebastian and Vulic, Ivan and S{\o}gaard, Anders},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={14},
  pages={12884--12892},
  year={2021}
}
```

