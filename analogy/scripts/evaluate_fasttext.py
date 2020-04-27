import os
import sys
import argparse
import logging
import yaml

from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from poutyne.framework import Model

from analogy.data import build_analogy_examples_from_file


LANGUAGES = [
    'da',
    'de',
    'en',
    'es',
    'fi',
    'fr',
    'it',
    'nl',
    'pl',
    'pt',
    'sv',
]


def process_entity(entity):
    elements = list()
    entity = entity.lower()
    es = entity.split()
    for e in es:
        elements.append(e)
    return elements


def preprocess_analogies(analogies):
    logging.info("Preprocessing analogies")
    processed_analogies = list()
    for analogy in analogies:
        processed_analogies.append(
            (process_entity(analogy.q_1_source),
            process_entity(analogy.q_1_target),
            process_entity(analogy.q_2_source),
            process_entity(analogy.q_2_target))
        )
    return processed_analogies


def build_vocab(processed_analogies):
    logging.info("Building vocabulary")
    vocab = set()
    for a in processed_analogies:
        for element in a:
            for e in element:
                vocab.add(e)
    return vocab


def get_vectors_for_vocab(path, vocab):
    logging.info("Getting word vectors for vocab")
    vectors = dict()
    with open(path, 'r', encoding='utf-8') as fhandle:
        for i, line in enumerate(fhandle):
            elements = line.split()
            if len(elements) > 2:
                try:
                    word = elements[0].lower()
                    if word in vocab:
                        vector = np.asarray([float(i) for i in elements[1:]])
                        vectors[word] = vector
                except:
                    pass
#                     print("Could not process line {}".format(i))
    return vectors


def build_word_mapping(vocab):
    logging.info("Building word to index mapping")
    word_to_idx = dict()
    for word in vocab:
        word_to_idx[word] = len(word_to_idx)
    return word_to_idx



class MyEmbeddings(nn.Embedding):
    def __init__(self, word_to_idx, embedding_dim):
        super(MyEmbeddings, self).__init__(len(word_to_idx), embedding_dim)
        self.embedding_dim = embedding_dim
        self.vocab_size = len(word_to_idx)
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}

    def set_item_embedding(self, idx, embedding):
        if len(embedding) == self.embedding_dim:
            self.weight.data[idx] = torch.FloatTensor(embedding)

    def load_words_embeddings(self, vec_model):
        logging.info("Loading word vectors in model")
        for word in vec_model:
            if word in self.word_to_idx:
                idx = self.word_to_idx[word]
                embedding = vec_model[word]
                self.set_item_embedding(idx, embedding)


class AnalogyModel(nn.Module):
    def __init__(self, embeddings):
        super(AnalogyModel, self).__init__()
        self.embeddings = embeddings
        self.loss = nn.CosineEmbeddingLoss()

    def loss_function(self, x, y):
        e1, e2, e3, e4, offset_trick, filters = x
        return self.loss(offset_trick, self.embeddings(e3), y)

    def accuracy(self, x, y):
        e1, e2, e3, e4, offset_trick, filters = x
        scores = offset_trick.matmul(self.embeddings.weight.transpose(0,1))
        return ((scores.argsort(descending=True)[:, :4] - e3.view(-1, 1).expand(len(e1), 4)) == 0).float().sum() / len(e1)

    def forward(self, inputs):
        e1 = inputs[:, 0]
        e2 = inputs[:, 1]
        e3 = inputs[:, 2]
        e4 = inputs[:, 3]
        e1_embeddings = self.embeddings(e1)
        e2_embeddings = self.embeddings(e2)
        e3_embeddings = self.embeddings(e3)
        e4_embeddings = self.embeddings(e4)
        offset_trick = e1_embeddings - e2_embeddings + e4_embeddings
        filters = torch.cat([e1.view(-1, 1), e2.view(-1, 1), e4.view(-1, 1)], dim=1)
        return e1, e2, e3, e4, offset_trick, filters


def vectorize_dataset(analogies, word_to_idx):
    elements = list()
    for e1, e2, e3, e4 in analogies:
        v_e1 = word_to_idx[e1]
        v_e2 = word_to_idx[e2]
        v_e3 = word_to_idx[e3]
        v_e4 = word_to_idx[e4]
        data = {
            "original": (e1, e2, e3, e4),
            "input_ids": (v_e1, v_e2, v_e3, v_e4)
        }
        elements.append(data)
    return elements


def merge_entity(entity, vectors):
    # This method merges an entity tokens
    # and create a new mean embedding from the token found in the entity
    new_entities = list()
    num_entities_found_in_vectors = 0
    new_vector = np.zeros_like(list(vectors.values())[0])
    for e in entity:
        if e in vectors:
            num_entities_found_in_vectors += 1
            new_entities.append(e)
            new_vector += vectors[e]
        else:
            new_entities.append("#UNK({})".format(e))
    new_entity = "_".join(new_entities)
    if num_entities_found_in_vectors > 0:
        if new_entity not in vectors:
            vectors[new_entity] = new_vector / num_entities_found_in_vectors  # Compute the mean for these
        return new_entity
    else:
        return None


def merge_entities_and_augment_vectors(analogies, vectors):
    logging.info("Merging entities and averaging their embeddings")
    analogies_with_merged_entities = list()
    for e1, e2, e3, e4 in tqdm(analogies):
        new_e1 = merge_entity(e1, vectors)
        new_e2 = merge_entity(e2, vectors)
        new_e3 = merge_entity(e3, vectors)
        new_e4 = merge_entity(e4, vectors)
        if new_e1 and new_e2 and new_e3 and new_e4:  # if we found anything for each entity
            analogies_with_merged_entities.append((
                new_e1,
                new_e2,
                new_e3,
                new_e4,
            ))
    return analogies_with_merged_entities


def collate(examples):
    input_ids = list()
    for e in examples:
        input_ids.append(e['input_ids'])
    return torch.tensor(input_ids), torch.LongTensor([1] * len(examples))


def evaluate(configs, language):
    cuda_device = 0
    device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")
    logging.info("Sending model to device {}".format(device))

    logging.info("Working on language {}".format(language))
    dataset_path = "./data/analogy_qids/analogy_{dataset}_{language}.csv".format(
        dataset=configs['dataset'],
        language=language
    )
    analogies = build_analogy_examples_from_file(dataset_path)
    if configs['test']:
        analogies = list(analogies)[:10000]

    processed_analogies = preprocess_analogies(analogies)
    vocab = build_vocab(processed_analogies)
    vector_path = './data/embeddings/wiki.{}.align.vec'.format(language)
    vectors = get_vectors_for_vocab(vector_path, vocab)
    analogies_with_merged_entities = merge_entities_and_augment_vectors(processed_analogies, vectors)
    word_to_idx = build_word_mapping(vectors.keys())
    vectorized_dataset = vectorize_dataset(analogies_with_merged_entities, word_to_idx)
    logging.info("Dataset size: {}".format(len(vectorized_dataset)))

    dataloader = DataLoader(vectorized_dataset, batch_size=128, collate_fn=collate)

    logging.info("Launching evaluation")
    my_embeddings = MyEmbeddings(word_to_idx, embedding_dim=300)
    my_embeddings.load_words_embeddings(vectors)
    model = AnalogyModel(my_embeddings)
    poutyne_model = Model(model, 'adam', loss_function=model.loss_function, batch_metrics=[model.accuracy])
    poutyne_model.to(device)
    loss, acc = poutyne_model.evaluate_generator(dataloader)
    logging.info("Accuracy: {}".format(acc))

    logging.info("Launching train")
    poutyne_model.fit_generator(dataloader)
    loss, acc = poutyne_model.evaluate_generator(dataloader)
    logging.info("Accuracy: {}".format(acc))


def main(configs):
    for language in LANGUAGES:
        evaluate(configs, language)


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    base_configs = yaml.load(open('./configs/base.yaml'), Loader=yaml.FullLoader)
    argument_parser = argparse.ArgumentParser()
    for config, value in base_configs.items():
        if type(value) is bool:
            # Hack as per https://stackoverflow.com/a/46951029
            argument_parser.add_argument('--{}'.format(config),
                                         type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                                         default=value)
        else:
            argument_parser.add_argument('--{}'.format(config), type=type(value), default=value)
    options = argument_parser.parse_args()
    configs = vars(options)
    main(configs)
