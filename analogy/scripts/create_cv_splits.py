import argparse
import os
import logging
import random

import numpy as np
from tqdm import tqdm

from analogy.data import build_analogy_examples_from_file


def filter_analogies(analogies):
    filtered_analogies = list()
    for a in analogies:
        if a.q_1_type == 'year' or a.q_2_type == 'year':
            continue
        else:
            filtered_analogies.append(a)
    return filtered_analogies


def write_analogies(analogies, fname, set_name, indices):
    filename = f"{fname}.{set_name}"
    logging.info(f"Creating {filename}")
    with open(filename, 'w') as fhandle:
        for index in indices:
            analogy = analogies[index]
            comment_row = analogy.get_comment_row()
            analogy_row = analogy.get_row()
            fhandle.write("{}\n".format(comment_row))
            fhandle.write("{}\n".format(analogy_row))


def main(args):
    data_path = args.data_path
    data_type = args.data_type
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    test_ratio = 1.0 - train_ratio - valid_ratio

    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    logging.info(f"Data path: {data_path}")
    logging.info(f"Output path: {data_path}")
    logging.info(f"Data type: {data_type}")
    logging.info(f"Train ratio: {train_ratio}")
    logging.info(f"Valid ratio: {valid_ratio}")
    logging.info(f"Test ratio: {test_ratio}")

    fnames = [f for f in os.listdir(data_path) if data_type in f]
    splits = dict()

    logging.info("Loading analogies for each language...")
    for fname in tqdm(fnames):
        file_path = os.path.join(data_path, fname)
        analogy_examples = build_analogy_examples_from_file(file_path)
        filtered_analogies = filter_analogies(analogy_examples)
        splits[fname] = list(filtered_analogies)
    logging.info("Done.")

    dataset_length = len(splits[fnames[0]])
    logging.info(f"Dataset length: {dataset_length}")

    indices = np.random.permutation(dataset_length)
    train_indices = indices[:int(dataset_length*train_ratio)]
    valid_indices = indices[int(dataset_length*train_ratio):int(dataset_length*(train_ratio+valid_ratio))]
    test_indices = indices[int(dataset_length*(train_ratio+valid_ratio)):]

    for fname, dataset in splits.items():
        file_path = os.path.join(output_path, fname)
        write_analogies(dataset, file_path, 'train', train_indices)
        write_analogies(dataset, file_path, 'valid', valid_indices)
        write_analogies(dataset, file_path, 'test', test_indices)




if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser(
          description='Split analogy data across languages')
    parser.add_argument('--data_path', type=str, help="Data directory", default='/home/mareike/PycharmProjects/analogies/data')
    parser.add_argument('--data_type', type=str, help="Either all or unique", default='all')
    parser.add_argument('--output_path', type=str, help="Output path", default='/home/mareike/PycharmProjects/analogies/output')
    parser.add_argument('--train_ratio', type=float, help="Ratio of training examples", default=0.6)
    parser.add_argument('--valid_ratio', type=float, help="Ratio of training examples", default=0.2)
    args = parser.parse_args()
    main(args)
