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
    analogy_reader = AnalogyReader(args.context)
    test_data = AnalogyDataset(analogy_reader.get_examples(os.path.join(args.data_path, args.test_data)), model=model)

    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    tokenizer = model._first_module().tokenizer
    evaluator = AnalogyEvaluator(test_dataloader, write_predictions=True, tokenizer=tokenizer, distance_file=args.distance_file, test_file=os.path.join(args.data_path,  args.test_data))

    model.evaluate(evaluator=evaluator, output_path= output_path)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(
          description='Train SentenceBert with analogy data')

    parser.add_argument('--model', type=str, default='37ea3401e1c94322ba01b9773ab1303f/',
                        help="The model to be loaded")
    parser.add_argument('--data_path', type=str,
                        help="Data directory", default='/home/mareike/PycharmProjects/analogies/data')
    parser.add_argument('--test_data', type=str,
                        help="csv file with analogies", default='analogy_unique_da_dists.csv.test.small')
    parser.add_argument('--distance_file', type=str, default='../../../data/analogy_unique_da_dists.csv',
                        help="Data file with distances for all analogies in the test set")
    parser.add_argument('--out', type=str,
                        help="output path", default='')
    parser.add_argument('--bs', type=int, default=16,
                        help="Batch size")
    parser.add_argument('--context', type=int, default=1,
                        help="Wether to use contextual information of analogies or not")





    args = parser.parse_args()
    main(args)
