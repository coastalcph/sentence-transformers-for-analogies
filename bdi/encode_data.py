import random
import numpy as np
import torch
from sentence_transformers import models
from sentence_transformers import test_config
from sentence_transformers.models import SmallBERT, BERT, XLMRoBERTa

from sentence_transformers import SentenceTransformer
from analogy.data import read_analogy_data, is_comment
import os
import argparse
import uuid

from sentence_transformers.util import bool_flag

"""
produce fasttext-style embedding files for analogy and word embedding data
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
    if args.finetuned_model:
        model = SentenceTransformer(args.finetuned_model)
    else:
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
    if args.data_type == 'word_embeddings':
        out_separator = ' '
        toks = []
        c = 0
        with open(args.data) as f:
            for line in f:
                if len(line.split(' ')) == 2:
                    continue
                c += 1
                if c %1000 == 0: print('read {} examples'.format(c))
                if c > args.max_data: break
                else: toks.append(line.split(' ')[0])
        print('Encoding... ')
        encoded_tokens = model.encode(toks)
        out_data = []
        print('Wrting embeddings to file')
        for i, tok in enumerate(toks):
            encoding = encoded_tokens[i]
            str_rep_encoding = ' '.join([str(elm) for elm in list(encoding)])
            out_data.append([tok, str_rep_encoding])

    elif args.data_type == 'analogies':
        out_separator = '\t'
        qids = []
        toks = []
        seen_toks = set()
        for row in read_analogy_data(args.data):
            if not is_comment(row):
                for e in ['Q1', 'Q2', 'Q3', 'Q4']:
                    qid = row['{}_id'.format(e)]
                    entity = row[e]
                    if entity not in seen_toks:
                        seen_toks.add(entity)
                        qids.append(qid)
                        toks.append(entity)

        encoded_tokens = model.encode(toks)
        out_data = []
        for i, qid in enumerate(qids):
            encoding = encoded_tokens[i]
            str_rep_encoding = ' '.join([str(elm) for elm in list(encoding)])
            out_data.append([qid, toks[i], str_rep_encoding])

    with open(os.path.join(args.out, args.outname), 'w') as fout:
        emb_dim = len(out_data[0][-1].split(' '))
        # first line is num_data emb_dim
        fout.write('{} {}\n'.format(len(out_data), emb_dim))
        for elm in out_data:
            fout.write('{}\n'.format('{}'.format(out_separator).join(elm)))
    fout.close()





if __name__ == '__main__':

    parser = argparse.ArgumentParser(
          description='Train SentenceBert with analogy data')

    parser.add_argument('--encoder', type=str,
                        default='small_bert',
                        choices=['bert-base-multilingual-cased', 'bert-base-uncased', 'small_bert', 'xlm-roberta-base'],
                        help="The pre-trained encoder used to encode the entities of the analogy")

    parser.add_argument('--finetuned_model', type=str,
                        default=None,
                        help="The finetuned model, e.g. trained on analogies, to be evaluated")
    parser.add_argument('--data_type', type=str,
                        default='word_embeddings',
                        choices=['word_embeddings', 'analogies'],
                        help="The data for which embeddings are computed. either analogies or words in fasttext word embedding files.")
    parser.add_argument('--data', type=str,
                        help="bilingual dictionary",
                        #default='/home/mareike/PycharmProjects/analogies/data/analogy_unique_en_contexts.csv.small')
                        default='/home/mareike/PycharmProjects/analogies/data/wiki.en.vec_1000')
    parser.add_argument('--out', type=str,
                        help="output path", default='encoded_toks')
    parser.add_argument('--outname', type=str,
                        help="name of output file", default='encodings.txt')
    parser.add_argument('--bs', type=int, default=8,
                        help="Batch size")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed")
    parser.add_argument('--max_data', type=int, default=300000,
                        help="maximum number of examples to encoded")

    args = parser.parse_args()

    main(args)