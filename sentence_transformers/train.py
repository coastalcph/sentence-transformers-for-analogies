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

    # Set up encoder
    if args.encoder == 'small_bert':
        # small BERT is a toy model
        word_embedding_model = SmallBERT(model_name_or_path='bert-base-uncased', config_dict=test_config.test_config)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)
    else:
        word_embedding_model = BERT(model_name_or_path=args.encoder)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Set up loss
    if args.loss == 'mse':
        train_loss = losses.AnalogyMSELoss(model=model)
    elif args.loss == 'hardtriplet':
        train_loss = losses.AnalogyBatchHardTripletLoss(sentence_embedder=model)


    # Load data
    analogy_reader = AnalogyReader()
    train_data = AnalogyDataset(analogy_reader.get_examples(os.path.join(args.data_path, args.train_data)), model=model)
    train_dataloader = DataLoader(train_data, shuffle=False, batch_size=batch_size)

    analogy_reader = AnalogyReader()
    dev_data = AnalogyDataset(analogy_reader.get_examples(os.path.join(args.data_path, args.dev_data)), model=model)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
    evaluator = AnalogyEvaluator(dev_dataloader)



    # Train
    model.fit(train_objectives=[(train_dataloader, train_loss)],
         evaluator=evaluator,
         epochs=args.epochs,
         evaluation_steps=5,
         warmup_steps=0,
         output_path=output_path
         )




if __name__ == '__main__':

    parser = argparse.ArgumentParser(
          description='Train SentenceBert with analogy data')

    parser.add_argument('--loss', type=str, default='hardtriplet',
                        choices = ['mse', 'hardtriplet'],
                        help="The loss function used")
    parser.add_argument('--encoder', type=str,
                        default='small_bert',
                        choices=['mbert', 'bert-base-uncased', 'small_bert'],
                        help="The pre-trained encoder used to encode the entities of the analogy")
    parser.add_argument('--data_path', type=str,
                        help="Data directory", default='/home/mareike/PycharmProjects/analogies/data')
    parser.add_argument('--train_data', type=str,
                        help="csv file with analogies", default='analogy_unique_en.csv.small')
    parser.add_argument('--dev_data', type=str,
                        help="csv file with analogies", default='analogy_unique_en.csv.dev')
    parser.add_argument('--out', type=str,
                        help="output path", default='')
    parser.add_argument('--bs', type=int, default=8,
                        help="Batch size")
    parser.add_argument('--epochs', type=int, default=5,
                        help="Number of training epochs")


    args = parser.parse_args()
    main(args)
