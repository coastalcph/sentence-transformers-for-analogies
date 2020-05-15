from . import SentenceEvaluator, SimilarityFunction
from torch.utils.data import DataLoader
import os
import torch
import logging
from tqdm import tqdm
from ..util import batch_to_device, combine_anchor_entities
import csv
from transformers import BertModel, BertTokenizer
from ..correlation_evaluation import read_analogies, read_dists
from scipy.stats import pearsonr

import numpy as np

class AnalogyEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on its ability to complete analogies.
    For each analogy, compute the closest entity to (e1-e2+e4) and check if its e3

    Returned score is accuracy over the whole dataset

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """


    def __init__(self, dataloader: DataLoader, main_similarity: SimilarityFunction = None, name: str = '', show_progress_bar: bool = None, write_predictions: bool=False, tokenizer: BertTokenizer=None, distance_file=None, test_file=None):
        """
        Constructs an evaluator based for the dataset

        :param dataloader:
            the data for the evaluation
        :param main_similarity:
            the similarity metric that will be used for the returned score
        """
        self.dataloader = dataloader
        self.main_similarity = main_similarity
        self.name = name
        if name:
            name = "_"+name

        if show_progress_bar is None:
            show_progress_bar = (logging.getLogger().getEffectiveLevel() == logging.INFO or logging.getLogger().getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.csv_file = "analogy_accuracy_evaluation"+name+"_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy", "candidates", "pearson"]
        self.write_predictions=write_predictions
        self.tokenizer = tokenizer

        if distance_file != None and test_file != None:
            self.analogy2dist = read_dists(distance_file)
            self.test_analogies = read_analogies(test_file)


    def __call__(self, model: 'SequentialSentenceEmbedder', output_path: str = None, epoch: int = -1,
                 steps: int = -1) -> float:
        """
        read from external file, all entities are candidates
        :param model:
        :param output_path:
        :param epoch:
        :param steps:
        :return:
        """
        model.eval()
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logging.info("Evaluation the model on " + self.name + " dataset" + out_txt)

        self.dataloader.collate_fn = model.smart_batching_collate

        iterator = self.dataloader
        if self.show_progress_bar:
            iterator = tqdm(iterator, desc="Convert Evaluating")

        rep_ea = []
        rep_candidates = []
        analogies2ids = []
        candidate2id = {}
        id2candidate = {}
        analogies = []
        # encode all entities of all analogies

        for step, batch in enumerate(iterator):
            features, label_ids = batch_to_device(batch, self.device)
            with torch.no_grad():
                reps = [model(sent_features)['sentence_embedding'] for sent_features in features]
                ea, _ = combine_anchor_entities(reps[0], reps[1], reps[2], reps[3])
                # candidates are all single entities in the dataset. check if they are already encoded

                bs = ea.shape[0]
                analogy_batch = [[] for x in range(bs)]
                analogy_ids = []
                for eid, sent_features in enumerate(features):
                    input_ids = sent_features['input_ids']
                    # iterate through batch
                    for i, input in enumerate(input_ids):
                        surface = self.tokenizer.convert_ids_to_tokens(input, skip_special_tokens=True)
                        analogy_batch[i].append(' '.join(surface))

                # for each analogy, add all entities to the candidate set if not present yet
                assert len(analogy_batch) == len(reps[0])
                for i, es in enumerate(analogy_batch):

                    def _add_to_candidate_set(emb_dim, analogy_tok_rep, analogy_emb_rep, rep_candidates, id2candidate,candidate2id):
                        assert len(rep_candidates) == len(id2candidate) == len(candidate2id)
                        for i in [0, 1, 2, 3]:
                            e = analogy_tok_rep[i]
                            rep = analogy_emb_rep[i].reshape(1, emb_dim)
                            if e not in candidate2id:
                                candidate2id[e] = len(rep_candidates)
                                rep_candidates.append(rep)
                                id2candidate[candidate2id[e]] = e
                        return rep_candidates, id2candidate, candidate2id

                    emb_rep = [reps[0][i], reps[1][i], reps[2][i], reps[3][i]]
                    rep_candidates, id2candidate, candidate2id = _add_to_candidate_set(emb_dim=ea.shape[1],analogy_tok_rep=es, analogy_emb_rep=emb_rep, rep_candidates=rep_candidates, id2candidate=id2candidate, candidate2id=candidate2id)
                    analogy_ids.append([candidate2id[es[0]], candidate2id[es[1]], candidate2id[es[2]], candidate2id[es[3]]])
                    assert id2candidate[analogy_ids[-1][2]] == es[2]
                rep_ea.append(ea)

                analogies.extend(analogy_batch)
                analogies2ids.extend(analogy_ids)

        rep_ea = torch.cat(rep_ea, 0)

        rep_candidates = torch.cat(rep_candidates, 0)
        assert rep_candidates.shape[0] == len(id2candidate) == len(candidate2id)
        print(rep_candidates.shape)
        ################################
        ####### Compute cosines sims ####
        #################################
        num_data = rep_ea.shape[0]
        a_norm = rep_ea / rep_ea.norm(dim=1)[:, None]
        b_norm = rep_candidates / rep_candidates.norm(dim=1)[:, None]
        cosine_sims = torch.mm(a_norm, b_norm.transpose(0, 1))

        top_ten_idxs = cosine_sims.argsort(descending=True)[:, :10]
        top4_idxs = top_ten_idxs[:, :4]

        def is_success(e3, e1_e2_e4, top4):
            if e3 not in top4:
                print('{} not in top4'.format(e3))
                return False
            else:
                for elem in top4:
                    if elem != e3 and elem not in e1_e2_e4:
                        return False
                    if elem == e3:
                        return True
        successes = 0
        d = -1
        for analogy, top4 in zip(analogies2ids, top4_idxs):
            d += 1
            analogy_str = '--'.join(['{}:::{}'.format(analogies[d][i], candidate2id[analogies[d][i]]) for i in range(4)])
            print('Analogy {}: {}'.format(d, analogy_str))
            top4 = top4.cpu().numpy()
            if is_success(analogy[2], {analogy[0], analogy[1], analogy[3]}, top4):
                successes += 1
                print('Success: {}\n'.format([id2candidate[pid] for pid in top4]))
            else:
                print('Fail: {}\n'.format([id2candidate[pid] for pid in top4]))
        print('Successes: {}, num_data {}'.format(successes, num_data))
        accuracy = successes/num_data

        logging.info("Accuracy:\t{:4f}".format(accuracy))
        num_candidates = rep_candidates.shape[0]

        if self.analogy2dist != None:
            assert len(self.test_analogies) == len(analogies2ids)
            # compute correlations between cos(ea, e3) and dist_biggraph(e1, e2, e3, e4)
            biggraph_dists = []
            cos_sims_model = []
            for aid, test_analogy in enumerate(self.test_analogies):
                # not all analogies have distances
                if test_analogy in self.analogy2dist:
                    biggraph_dists.append(self.analogy2dist[test_analogy])
                    # get the cosine sim that the model calculated
                    idx_e3 = analogies2ids[aid][2]
                    cos_sims_model.append(cosine_sims[aid, idx_e3].item())
            pearson = pearsonr(biggraph_dists, cos_sims_model)[0]
        else: pearson = None
        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, accuracy, num_candidates, pearson])
            else:
                with open(csv_path, mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, accuracy, num_candidates, pearson])


        if self.write_predictions:

            assert len(analogies) == cosine_sims.shape[0]
            with open(os.path.join(output_path, 'predictions_{}.csv'.format(0)), 'w') as fpred:
                pred_writer = csv.writer(fpred, delimiter=';')
                # write out the model predictions
                top_ten = top_ten_idxs.cpu().numpy()
                for aid in range(cosine_sims.shape[0]):
                    predictions = [id2candidate[pid] for pid in top_ten[aid]]
                    e1, e2, e3, e4 = analogies2ids[aid][0], analogies2ids[aid][1], analogies2ids[aid][2], analogies2ids[aid][3]
                    success = int(is_success(e3, {e1, e2, e4}, top_ten[aid][:4]))
                    pred_writer.writerow([analogies[aid][0], analogies[aid][1], analogies[aid][2], analogies[aid][3],
                                          ','.join(predictions), success])
            fpred.close()
        return accuracy

