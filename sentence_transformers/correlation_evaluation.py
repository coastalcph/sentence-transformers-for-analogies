import configparser
import os
import csv
from analogy.data import is_comment
import itertools
from scipy.stats import pearsonr
import argparse
import numpy as np

def read_dists(dist_file):
    with open(dist_file, newline='') as csvfile:
        fieldnames = ['Q1', 'Q1_id', 'Q2', 'Q2_id', 'Q3', 'Q3_id', 'Q4', 'Q4_id', 'all_dist', 'pairwise_dist']
        reader = csv.DictReader(csvfile, delimiter=';', fieldnames=fieldnames)
        analogy2dist = {}
        for row in reader:
            if not is_comment(row):
                if not row['pairwise_dist'] == '':
                    analogy2dist[(row['Q1_id'], row['Q2_id'], row['Q3_id'], row['Q4_id'])] = float(row['pairwise_dist'])
    return analogy2dist

def read_analogies(fname):
    with open(fname, newline='') as csvfile:
        fieldnames = ['Q1', 'Q1_id', 'Q2', 'Q2_id', 'Q3', 'Q3_id', 'Q4', 'Q4_id', 'all_dist', 'pairwise_dist']
        reader = csv.DictReader(csvfile, delimiter=';', fieldnames=fieldnames)
        data = []
        for row in reader:
            if not is_comment(row):
                data.append((row['Q1_id'], row['Q2_id'], row['Q3_id'], row['Q4_id']))
    return data

def read_predictions(fname):
    with open(fname, newline='') as csvfile:
        fieldnames = ['Q1', 'Q2', 'Q3', 'Q4', 'predictions', 'success']
        reader = csv.DictReader(csvfile, delimiter=';', fieldnames=fieldnames)
        data = []
        for row in reader:
            if not is_comment(row):
                data.append(row)
    return data

def compare_prediction(prediction, analogy):
    for key in ['Q1', 'Q2', 'Q3', 'Q4']:
        def clean(s):
            return s.replace('#', '').replace(' ', '').lower()
        if clean(prediction[key]) != clean(analogy[key]):
            print(clean(prediction[key]))
            print(clean(analogy[key]))
    return True

def main(args):
 
    distance_file = args.distance_file
    predictions_file = args.predictions_file
    analogies_file = args.test_file


    analogy2dist = read_dists(distance_file)

    predictions = read_predictions(predictions_file)
    test_analogies = read_analogies(analogies_file)
    dists = []
    successes = []
    for aid, test_analogy in enumerate(test_analogies):
        # not all analogies have distances
        if test_analogy in analogy2dist:
            dists.append(analogy2dist[test_analogy])
            successes.append(int(predictions[aid]['success']))

    min_val = np.min(dists)
    max_val = np.max(dists)
    eff_intervals = [min_val]
    for elm in args.intervals:
        if elm > min_val and elm < max_val:
            eff_intervals.append(elm)

    eff_intervals.append(max_val)
    print(eff_intervals)
    sorting_idx = np.argsort(dists)
    sorted_dists = [dists[i] for i in sorting_idx]
    sorted_successes = [successes[i] for i in sorting_idx]
    bins = []
    bin = []
    thr_idx = 0
    accs = []
    for s, dist in zip(sorted_successes, sorted_dists):
        if dist > eff_intervals[thr_idx]:
            bins.append(bin)
            bin = [(s, dist)]
            thr_idx += 1
        else:
            bin.append((s, dist))
    with open(args.out_file, 'w') as f:
        wr = csv.writer(f, delimiter=';')
        wr.writerow([eff_intervals])
        for i, bin in enumerate(bins):
            tp = np.sum([elm[0] for elm in bin])
            acc = tp / len(bin)
            accs.append(acc)
            print('[{} {}) {}/{} acc: {}'.format(eff_intervals[i], eff_intervals[i + 1], tp, len(bin), acc))
            wr.writerow(['[{} {})'.format(eff_intervals[i], eff_intervals[i + 1]), '{}/{}'.format(tp, len(bin)), acc])
        all = list(itertools.chain.from_iterable(bins))
        tp = np.sum([elm[0] for elm in all])
        acc = tp / len(all)
        print('ALL {}/{} acc: {}'.format(tp, len(all), acc))
        wr.writerow(['ALL', '{}/{}'.format(tp, len(all)), acc])
        pearson = pearsonr(accs, eff_intervals[:-1])
        print('Pearson: {}'.format(pearson))
        wr.writerow(['Pearson', pearson])
    f.close()


if __name__=="__main__":


    parser = argparse.ArgumentParser(
        description='Train SentenceBert with analogy data')

    parser.add_argument('--distance_file', type=str, default='../../../data/analogy_all_en_dists.csv',
                        help="Data file with distances for all analogies in the test set")
    parser.add_argument('--predictions_file', type=str, default='e05d7ea57fdc4920957a3f5959c7a332/predictions_0.csv',
                        help="Data file with predictions")
    parser.add_argument('--test_file', type=str, default='../../../data/analogy_unique_en.csv.small',
                        help="Data file with test analogies")

    parser.add_argument('--intervals', type=list, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        help="Frequency intervals [)")
    parser.add_argument('--out_file', type=str, default='e05d7ea57fdc4920957a3f5959c7a332/frequency_bins.csv',
                        help="Data file with test analogies")


    args = parser.parse_args()
    main(args)
