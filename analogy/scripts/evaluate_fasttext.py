import os
import sys
import argparse
import logging
import yaml
from typing import List

from dataclasses import dataclass

from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from poutyne.framework import Model
from poutyne.framework.metrics import EpochMetric

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


@dataclass
class AnalogyExample:
    e1: List[str]
    e2: List[str]
    e3: List[str]
    e4: List[str]
    distance: float


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
        processed_analogies.append(AnalogyExample(
            process_entity(analogy.q_1_source),
            process_entity(analogy.q_1_target),
            process_entity(analogy.q_2_source),
            process_entity(analogy.q_2_target),
            float(analogy.distance)
        ))
    return processed_analogies


def build_vocab(processed_analogies):
    logging.info("Building vocabulary")
    vocab = set()
    for a in processed_analogies:
        for element in a.e1:
            vocab.add(element)
        for element in a.e2:
            vocab.add(element)
        for element in a.e3:
            vocab.add(element)
        for element in a.e4:
            vocab.add(element)
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


class CorrelationMetric(EpochMetric):
    def __init__(self) -> None:
        super().__init__()
        self.scores = list()
        self.distances = list()

    def forward(self, x, y):
        # Accumulate metrics here
        e1, e2, e3, e4, offset_trick, scores, distances = x
        for i, (s, d) in enumerate(zip(scores, distances)):
            self.scores.append(1 - float(s[e3[i]]))
            self.distances.append(float(d))  # We append the distance

    def get_metric(self):
        return np.corrcoef(self.scores, self.distances)[0][1]

    def reset(self) -> None:
        self.scores = list()
        self.distances = list()


class AnalogyModel(nn.Module):
    def __init__(self, embeddings):
        super(AnalogyModel, self).__init__()
        self.embeddings = embeddings
        self.loss = nn.CosineEmbeddingLoss()

    def loss_function(self, x, y):
        e1, e2, e3, e4, offset_trick, scores, distances = x
        return self.loss(offset_trick, self.embeddings(e3), y)

    def is_success(self, e3, e1_e2_e3, top4):
        return e3 in top4 and len(e1_e2_e3 & top4) == len(e1_e2_e3)

    def accuracy(self, x, y):
        e1s, e2s, e3s, e4s, offset_trick, scores, distances = x
        sorted_indexes_by_scores = scores.argsort(descending=True)[:, :4]
        accuracies = list()
        for e1, e2, e3, e4, top4_indexes in zip(e1s, e2s, e3s, e4s, sorted_indexes_by_scores):
            success = self.is_success(e3, {e1, e2, e4}, set(top4_indexes))
            if success:
                accuracies.append(1)
            else:
                accuracies.append(0)
        return sum(accuracies) / len(accuracies)

    def forward(self, input_ids, distances):
        e1 = input_ids[:, 0]
        e2 = input_ids[:, 1]
        e3 = input_ids[:, 2]
        e4 = input_ids[:, 3]
        e1_embeddings = self.embeddings(e1)
        e2_embeddings = self.embeddings(e2)
        e3_embeddings = self.embeddings(e3)
        e4_embeddings = self.embeddings(e4)
        offset_trick = e1_embeddings - e2_embeddings + e4_embeddings
        a_norm = offset_trick / offset_trick.norm(dim=1)[:, None]
        b_norm = self.embeddings.weight / self.embeddings.weight.norm(dim=1)[:, None]
        cosine_sims = torch.mm(a_norm, b_norm.transpose(0,1))
        return e1, e2, e3, e4, offset_trick, cosine_sims, distances


def vectorize_dataset(analogies, word_to_idx):
    elements = list()
    for a in analogies:
        v_e1 = word_to_idx[a.e1]
        v_e2 = word_to_idx[a.e2]
        v_e3 = word_to_idx[a.e3]
        v_e4 = word_to_idx[a.e4]
        data = {
            "original": (a.e1, a.e2, a.e3, a.e4),
            "input_ids": (v_e1, v_e2, v_e3, v_e4),
            "distance": a.distance
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
    for a in tqdm(analogies):
        new_e1 = merge_entity(a.e1, vectors)
        new_e2 = merge_entity(a.e2, vectors)
        new_e3 = merge_entity(a.e3, vectors)
        new_e4 = merge_entity(a.e4, vectors)
        if new_e1 and new_e2 and new_e3 and new_e4:  # if we found anything for each entity
            analogies_with_merged_entities.append(AnalogyExample(
                new_e1,
                new_e2,
                new_e3,
                new_e4,
                a.distance
            ))
    return analogies_with_merged_entities


def collate(examples):
    input_ids = list()
    distances = list()
    for e in examples:
        input_ids.append(e['input_ids'])
        distances.append(e['distance'])
    return (torch.tensor(input_ids), torch.FloatTensor(distances)), torch.LongTensor([1] * len(examples))


def evaluate(configs, language):
    cuda_device = 0
    device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")
    logging.info("Sending model to device {}".format(device))

    logging.info("Working on language {}".format(language))
    dataset_path = "./data/analogy_dists/analogy_{dataset}_{language}_dists.csv".format(
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
    poutyne_model = Model(model, 'adam', loss_function=model.loss_function, batch_metrics=[model.accuracy], epoch_metrics=[CorrelationMetric()])
    poutyne_model.to(device)
    loss, acc = poutyne_model.evaluate_generator(dataloader)
    logging.info("Accuracy: {}".format(acc[0]))
    logging.info("Correlation: {}".format(acc[1]))

    # logging.info("Launching train")
    # poutyne_model.fit_generator(dataloader)
    # loss, acc = poutyne_model.evaluate_generator(dataloader)
    # logging.info("Accuracy: {}".format(acc))


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
