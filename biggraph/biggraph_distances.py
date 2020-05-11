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
from analogy.data import read_analogy_data, is_comment


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
                distances = compute_biggraph_distances_pair(q1, q2, emb_file, pointers)
                if distances:
                    d_all = distances[0][0]
                    d_pairwise = None
                else:
                    d_all = None
                    d_pairwise = None
            else:
                q1, q2, q3, q4 = row['Q1_id'], row['Q2_id'], row['Q3_id'], row['Q4_id']
                d_all, d_pairwise = compute_biggraph_distances_quadruplet(q1, q2, q3, q4, emb_file, pointers)
                
            outrow = [row['Q1'], row['Q1_id'], row['Q2'], row['Q2_id'], row['Q3'], row['Q3_id'], row['Q4'], row['Q4_id'], d_all, d_pairwise]
            writer.writerow(outrow)
    f.close()


def compute_biggraph_distances_pair(qid1, qid2, emb_file, pointers):
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


def compute_biggraph_distances_quadruplet(qid1, qid2, qid3, qid4, emb_file, pointers,setting="pairwise"):
    """
    compute distances between elements of the analogy. Returns a a tuple of distances, 1. between all possible pairs in the analogy
    and 2. d= (cos(e1,e2) + cos(e3,e4))/2
    :param qid1:
    :param qid2:
    :param qid3:
    :param qid4:
    :param emb_file:
    :param pointers:
    :param setting: if pairwise, distances are computed as
    :return:
    """
    if qid1 in pointers and qid2 in pointers and qid3 in pointers and qid4 in pointers:
        emb1 = graphreader.get_embedding(emb_file, pointers[qid1])
        emb2 = graphreader.get_embedding(emb_file, pointers[qid2])
        emb3 = graphreader.get_embedding(emb_file, pointers[qid3])
        emb4 = graphreader.get_embedding(emb_file, pointers[qid4])
        embs = np.array([emb1, emb2, emb3, emb4])
        dists = cosine_distances(embs, embs)
        d_all = np.sum(dists)/(dists.shape[0]*(dists.shape[0]-1))
        d_pairwise = (dists[0,1] + dists[2,3])/2
        return d_all, d_pairwise
    if qid1 not in pointers:
        print('{} not in embeddings'.format(qid1))
    if qid2 not in pointers:
        print('{} not in embeddings'.format(qid2))
    if qid3 not in pointers:
        print('{} not in embeddings'.format(qid3))
    if qid4 not in pointers:
        print('{} not in embeddings'.format(qid4))
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

