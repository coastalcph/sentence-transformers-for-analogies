"""
add distances between entity representations from the wikidata biggraph embeddings to the analogy quadruplets
"""
import csv
import re
import os
import configparser
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from biggraph import graphreader
from data import read_analogy_data, is_comment


def augment_analogy_data(fname_in, fname_out, emb_file, pointers):
    """
    augment analogy data with distances computed over the wikidata biggraph embeddings for the entities
    :param fname:
    :param emb_file:
    :param pointers:
    :return:
    """
    q_pattern = re.compile('#.*?\((Q[0-9]+)\).*\((P[0-9]+)\).*\((Q[0-9]+)\)')
    with open(fname_out, 'w') as f:
        writer = csv.writer(f, delimiter=';')
        for row in read_analogy_data(fname_in):
            if is_comment(row):
                # get distances
                match = re.search(q_pattern, row['Q1'])
                q1, p, q2 = match.group(1), match.group(2), match.group(3)
                distances = compute_biggraph_distances(q1, q2, emb_file, pointers)
                if distances:
                    d = distances[0][0]
                else:
                    d = None
            outrow = [row['Q1'], row['Q2'], row['Q3'], row['Q4'], d]
            writer.writerow(outrow)
    f.close()


def compute_biggraph_distances(qid1, qid2, emb_file, pointers):
    if qid1 in pointers and qid2 in pointers:
        emb1 = graphreader.get_embedding(emb_file, pointers[qid1])
        emb2 = graphreader.get_embedding(emb_file, pointers[qid2])
        dists = cosine_distances(np.array(emb1).reshape(1, -1), np.array(emb2).reshape(1, -1))
        return dists
    if qid1 not in pointers:
        print('{} not in embeddings'.format(qid1))
    if qid2 not in pointers:
        print('{} not in embeddings'.format(qid2))
    return None

if __name__=="__main__":

    cfg = 'config.cfg'
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(cfg)

    embeddings_path = config.get('Files', 'biggraph_embs')
    pointers_path = config.get('Files', 'biggraph_pointers')
    emb_file = graphreader.file_open(embeddings_path)
    pointers = graphreader.load_pointers(pointers_path)
    langs = ['da', 'de', 'en', 'es', 'fi', 'fr', 'it', 'nl', 'pl', 'pt', 'sv']
    for lang in langs:
        fname = os.path.join(config.get('Files', 'data'), 'analogy_all_{}.csv'.format(lang))
        fname_out = os.path.join(config.get('Files', 'data'), 'analogy_all_{}_dists.csv'.format(lang))
        augment_analogy_data(fname, fname_out, emb_file, pointers)
