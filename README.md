# Sentence Transformers for Analogies

This is the code repository for the AAAI paper [Analogy Training Multilingual Encoders](https://ojs.aaai.org/index.php/AAAI/article/view/17524/17331) (Garneau et al. 2021). It is a fork from the original [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) repository.

The repository provides 
- the [analogy dataset](https://bit.ly/3aaKTzF) used to train the multilingual encoder
- the improved multilingual encoder after training on the analogy dataset
- scripts for training and evaluating models

## Dataset
## Model
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

