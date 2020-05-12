import os
import sys
import argparse
import logging
import yaml
from typing import List
import copy
import random

from dataclasses import dataclass

from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from poutyne.framework import Model

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import Normalizer

from analogy.data import build_analogy_examples_from_file
from analogy.metrics import CorrelationMetric, CorrelationBinnedAccuracyMetric
from analogy.models import MyEmbeddings, AnalogyModel, IdentityMapper, NeuralMapper


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


def r2_score_pytorch(y_pred, y_true):
    return r2_score(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())


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
            float(analogy.distance_pairwise)
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


def split_examples(dataset, ratio=0.8):
    np.random.shuffle(dataset)
    train = dataset[:int(len(dataset)*ratio)]
    test = dataset[int(len(dataset)*ratio):]
    return train, test


def filter_vectors_for_entities(analogies, vectors):
    entities_set = set()
    for a in analogies:
        entities_set.add(a.e1)
        entities_set.add(a.e2)
        entities_set.add(a.e3)
        entities_set.add(a.e4)
    filtered_vector_set = {k: v for k, v in vectors.items() if k in entities_set}
    return filtered_vector_set


def load_set(language, dataset_path, set_name, test=False):
    set_path = f"{dataset_path}.{set_name}"
    analogies = build_analogy_examples_from_file(set_path)
    if test:
        analogies = list(analogies)[:10000]
    processed_analogies = preprocess_analogies(analogies)
    vocab = build_vocab(processed_analogies)
    vector_path = './data/embeddings/wiki.{}.align.vec'.format(language)
    vectors = get_vectors_for_vocab(vector_path, vocab)
    analogies_with_merged_entities = merge_entities_and_augment_vectors(processed_analogies, vectors)
    filtered_vector_set = filter_vectors_for_entities(analogies_with_merged_entities, vectors)
    word_to_idx = build_word_mapping(vectors.keys())
    return analogies_with_merged_entities, filtered_vector_set, word_to_idx


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight.data)


def evaluate(configs, language):
    cuda_device = 0
    device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")
    logging.info("Sending model to device {}".format(device))

    logging.info("Working on language {}".format(language))
    dataset_path = "./data/analogy_dists_splits/analogy_{dataset}_{language}_dists.csv".format(
        dataset=configs['dataset'],
        language=language
    )

    train_set, train_vectors, train_word_to_idx = load_set(language, dataset_path, 'train', configs['test'])
    valid_set, valid_vectors, valid_word_to_idx = load_set(language, dataset_path, 'valid', configs['test'])
    test_set, test_vectors, test_word_to_idx = load_set(language, dataset_path, 'test', configs['test'])

    vectorized_train = vectorize_dataset(train_set, train_word_to_idx)
    vectorized_valid = vectorize_dataset(valid_set, valid_word_to_idx)
    vectorized_test = vectorize_dataset(test_set, test_word_to_idx)

    train_loader = DataLoader(vectorized_train, batch_size=128, collate_fn=collate, shuffle=True, drop_last=True)
    valid_loader = DataLoader(vectorized_valid, batch_size=128, collate_fn=collate)
    test_loader = DataLoader(vectorized_test, batch_size=128, collate_fn=collate)

    train_embeddings = MyEmbeddings(train_word_to_idx, embedding_dim=300)
    train_embeddings.load_words_embeddings(train_vectors)

    valid_embeddings = MyEmbeddings(valid_word_to_idx, embedding_dim=300)
    valid_embeddings.load_words_embeddings(valid_vectors)

    test_embeddings = MyEmbeddings(test_word_to_idx, embedding_dim=300)
    test_embeddings.load_words_embeddings(test_vectors)


    model = AnalogyModel(train_embeddings, valid_embeddings, test_embeddings, configs['reg_term_lambda'], configs['delta'])
    mapper = IdentityMapper()
    model.set_mapper(mapper)

    poutyne_model = Model(
        model,
        'adam',
        loss_function=model.loss_function,
        batch_metrics=[model.accuracy],
        epoch_metrics=[CorrelationMetric(), CorrelationBinnedAccuracyMetric()]
    )
    poutyne_model.to(device)

    loss, (acc, corr, _) = poutyne_model.evaluate_generator(valid_loader)
    logging.info("Statistics on valid set before train (IdentityMapper used);")
    logging.info("Accuracy: {}".format(acc))
    logging.info("Correlation: {}".format(corr))

    # Setting the embedding table
    model.test_embeddings = test_embeddings
    loss, (acc, corr, _) = poutyne_model.evaluate_generator(test_loader)
    logging.info("Statistics on test set before train (IdentityMapper used);")
    logging.info("Accuracy: {}".format(acc))
    logging.info("Correlation: {}".format(corr))

    logging.info("Launching train")
    poutyne_model.fit_generator(train_loader, epochs=10)

    # Train mapper
    original_embeddings = MyEmbeddings(train_word_to_idx, embedding_dim=300)
    original_embeddings.load_words_embeddings(train_vectors)
    normalizer = Normalizer()
    X = original_embeddings.weight.data.cpu().numpy()
    Y = train_embeddings.weight.data.cpu().numpy()
    X_norm = normalizer.transform(X)
    Y_norm = normalizer.transform(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X_norm, Y_norm)
    mapper_model = nn.Sequential(
        nn.Linear(300, 512),
        nn.Tanh(),
        nn.Linear(512, 512),
        nn.Tanh(),
        nn.Linear(512, 300),
    )
    mapper_model.apply(init_weights)
    poutyne_mapper_model = Model(mapper_model, 'adam', 'mse', batch_metrics=[r2_score_pytorch])
    poutyne_mapper_model.to(device)
    poutyne_mapper_model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=30,
        batch_size=64
    )

    neural_mapper = NeuralMapper(mapper_model, device)
    model.set_mapper(neural_mapper)

    model.test_embeddings = valid_embeddings
    loss, (acc, corr, _) = poutyne_model.evaluate_generator(valid_loader)
    logging.info("Statistics on valid set after train (IdentityMapper used);")
    logging.info("Accuracy: {}".format(acc))
    logging.info("Correlation: {}".format(corr))

    model.test_embeddings = test_embeddings
    loss, (acc, corr, _) = poutyne_model.evaluate_generator(test_loader)
    logging.info("Statistics on test set after train (NeuralMapper used);")
    logging.info("Accuracy: {}".format(acc))
    logging.info("Correlation: {}".format(corr))



def main(configs):
    for language in LANGUAGES:
        evaluate(configs, language)


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
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
