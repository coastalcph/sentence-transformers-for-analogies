from sentence_transformers import models
from sentence_transformers import test_config
from transformers import BertConfig, BertModel
from sentence_transformers.models import SmallBERT, BERT
from sentence_transformers import SentenceTransformer
from sentence_transformers.readers import NLIDataReader, STSDataReader, AnalogyReader
from sentence_transformers import losses
from torch.utils.data import DataLoader
from sentence_transformers.datasets import SentencesDataset, AnalogyDataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, AnalogyEvaluator
import os
import argparse
import uuid

def main(args):

    if args.out == '':
        output_path = uuid.uuid4().hex
    else:
        output_path = args.out
    batch_size = args.bs


    model = SentenceTransformer(args.model)


    # Load data
    analogy_reader = AnalogyReader()
    test_data = AnalogyDataset(analogy_reader.get_examples(os.path.join(args.data_path, args.test_data)), model=model)

    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    tokenizer = model._first_module().tokenizer
    evaluator = AnalogyEvaluator(test_dataloader, write_predictions=True, tokenizer=tokenizer)

    model.evaluate(evaluator=evaluator, output_path= output_path)







if __name__ == '__main__':

    parser = argparse.ArgumentParser(
          description='Train SentenceBert with analogy data')

    parser.add_argument('--model', type=str, default='e826f2d4bafa4f2c8713d6a2d70cbffd/',
                        help="The model to be loaded")
    parser.add_argument('--data_path', type=str,
                        help="Data directory", default='/home/mareike/PycharmProjects/analogies/data')
    parser.add_argument('--test_data', type=str,
                        help="csv file with analogies", default='analogy_unique_en.csv.small')
    parser.add_argument('--out', type=str,
                        help="output path", default='')
    parser.add_argument('--bs', type=int, default=16,
                        help="Batch size")



    args = parser.parse_args()
    main(args)
