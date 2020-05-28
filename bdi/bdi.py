import random
import numpy as np
import torch
from sentence_transformers import models
from sentence_transformers import test_config

from transformers import BertConfig, BertModel
from sentence_transformers.models import SmallBERT, BERT, XLMRoBERTa

from sentence_transformers import SentenceTransformer
from sentence_transformers.readers import BiDictReader
from sentence_transformers import losses
from torch.utils.data import DataLoader
from sentence_transformers.datasets import BDIDataset
from sentence_transformers.evaluation import BDIEvaluator
import os
import argparse
import uuid

from sentence_transformers.util import bool_flag

"""
do  BDI, i.e. encode words/MWEs in both languages using the encoder and retrieve NNs across languages
"""


def main(args):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.out == '':
        output_path = uuid.uuid4().hex
    else:
        output_path = args.out

    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        if os.listdir(output_path):
            raise ValueError("Output directory ({}) already exists and is not empty.".format(
                output_path))
    print('Writing log to {}'.format(output_path))
    batch_size = args.bs

    # Set up encoder
    if args.encoder == 'small_bert':
        # small BERT is a toy model
        word_embedding_model = SmallBERT(model_name_or_path='bert-base-uncased', config_dict=test_config.test_config, do_lower_case=False)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)
    elif args.encoder == 'xlm-roberta-base':
        word_embedding_model = XLMRoBERTa(model_name_or_path=args.encoder)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)
    else:

        word_embedding_model = BERT(model_name_or_path=args.encoder, do_lower_case=False)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


    # Load data
    bdi_reader = BiDictReader()
    # src - target pairs
    src_examples, trg_examples, src2trg = bdi_reader.get_examples(args.eval_data, args.candidates, sep=' ')
    for s, ts in src2trg.items():
        for t in ts:
            print('{} {}'.format(src_examples[s].texts, trg_examples[t].texts))


    src_words = [elm.texts[0] for elm in src_examples]
    trg_words = [elm.texts[0] for elm in trg_examples]

    eval = BDIEvaluator(dataloader=None, src_words=src_words, trg_words=trg_words, src2trg=src2trg)
    eval(model, output_path=output_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
          description='Train SentenceBert with analogy data')

    parser.add_argument('--encoder', type=str,
                        default='small_bert',
                        choices=['bert-base-multilingual-cased', 'bert-base-uncased', 'small_bert', 'xlm-roberta-base'],
                        help="The pre-trained encoder used to encode the entities of the analogy")
    parser.add_argument('--pretrained_model', type=str,
                        default=None,
                        help="The pre-trained model, e.g. trained on analogies,  to be evaluated")
    parser.add_argument('--data_path', type=str,
                        help="Data directory", default='/home/mareike/PycharmProjects/analogies/data')
    parser.add_argument('--candidates', type=str,
                        help="candidates added to the dev data if not present yet",
                        default='/home/mareike/PycharmProjects/analogies/data/bdi/muse/candidates/candidates.en.small')
    parser.add_argument('--eval_data', type=str,
                        help="bilingual dictionary",
                        default='/home/mareike/PycharmProjects/analogies/data/bdi/muse/dictionaries/de-en.5000-6500.txt')
    parser.add_argument('--out', type=str,
                        help="output path", default='')
    parser.add_argument('--bs', type=int, default=8,
                        help="Batch size")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    main(args)
