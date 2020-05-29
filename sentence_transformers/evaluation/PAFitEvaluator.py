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

class PAFitEvaluator(SentenceEvaluator):
    """
    Evaluates the  Procrustes fit of a model in a BDI task
    """


    def __init__(self, dataloader: DataLoader, main_similarity: SimilarityFunction = None, name: str = '', show_progress_bar: bool = None, write_predictions: bool=False, tokenizer: BertTokenizer=None,
                 src_words=None, trg_words=None, src2trg=None):
        """
        Constructs an evaluator for the dataset

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
        self.csv_file = "bdi_accuracy_evaluation"+name+"_results.csv"
        self.csv_headers = ["epoch", "steps", "p@1", "p@5", 'p@10', 'candidates']
        self.write_predictions=write_predictions
        self.tokenizer = tokenizer

        self.src_words = src_words
        self.trg_words = trg_words
        self.src2trg = src2trg


    def compute_PA(self, x, y):
        U, S, Vt = np.linalg.svd(torch.mm(torch.transpose(y, 0, 1), x), full_matrices=True)
        W = torch.from_numpy(U.dot(Vt))
        return W


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


        # encode the bilingual dictionaries
        bd_src_words = []
        bd_trg_words = []
        for sid, tids in self.src2trg.items():
            for tid in tids:
                bd_src_words.append(self.src_words[sid])
                bd_trg_words.append(self.trg_words[tid])
        bd_rep_src = torch.cat([torch.from_numpy(elm.reshape(1, elm.shape[0])) for elm in model.encode(bd_src_words)], 0)
        bd_rep_trg = torch.cat([torch.from_numpy(elm.reshape(1, elm.shape[0])) for elm in model.encode(bd_trg_words)],
                               0)
        mapping = self.compute_PA(bd_rep_src, bd_rep_trg)

        rep_src = torch.cat([torch.from_numpy(elm.reshape(1, elm.shape[0])) for elm in model.encode(self.src_words)], 0)
        rep_src = torch.mm(rep_src, mapping)
        rep_trg = torch.cat([torch.from_numpy(elm.reshape(1, elm.shape[0])) for elm in model.encode(self.trg_words)], 0)



        ################################
        ####### Compute cosines sims ####
        #################################
        num_data = rep_src.shape[0]
        a_norm = rep_src / rep_src.norm(dim=1)[:, None]
        b_norm = rep_trg / rep_trg.norm(dim=1)[:, None]
        cosine_sims = torch.mm(a_norm, b_norm.transpose(0, 1))

        top_n_idxs = cosine_sims.argsort(descending=True)

        p_at_10 = 0.
        p_at_5 = 0.
        p_at_1 = 0.

        def trg_in_top_n(correct_idxs, retrieved_idxs):
            ns = []
            for correct_idx in correct_idxs:
                for i, idx in enumerate(retrieved_idxs):
                    idx = idx.data.item()
                    if correct_idx == idx:
                        ns.append(i+1)
                        break
            if len(ns) == 0:
                return -1
            return np.min(ns)



        for sid, tids in self.src2trg.items():
            top_ten = top_n_idxs[sid][:10]
            top_retrieved_trgs = ','.join([self.trg_words[i.data.item()] for i in top_ten])
            n = trg_in_top_n(correct_idxs=tids, retrieved_idxs=top_ten)
            #print('Retrieved one of trgs _{}_ for src word _{}_ on rank {}'\
            #             .format(','.join([self.trg_words[i] for i in self.src2trg[sid]]), self.src_words[sid], n))
            #print('Top 10 for src words _{}_: {}\n'.format(self.src_words[sid], top_retrieved_trgs))
            logging.info('Retrieved one of trgs _{}_ for src word _{}_ on rank {}'\
                         .format(','.join([self.trg_words[i] for i in self.src2trg[sid]]), self.src_words[sid], n))
            logging.info('Top 10 for src words _{}_: {}'.format(self.src_words[sid], top_retrieved_trgs))
            if n == 1:
                p_at_10 += 1
                p_at_5 += 1
                p_at_1 += 1
            elif 0 < n <= 5:
                p_at_5 += 1
                p_at_10 += 1
            elif 0 < n <= 10:
                p_at_10 += 1
        print(p_at_1)
        print(p_at_5)
        print(p_at_10)
        p_at_10 = p_at_10/num_data
        p_at_5 = p_at_5/num_data
        p_at_1 = p_at_1 / num_data

        print("P@1:\t{:4f}\tP@5:\t{:4f}\tP@10:\t{:4f}".format(p_at_1,  p_at_5, p_at_10))
        logging.info("P@1:\t{:4f}\tP@5:\t{:4f}\tP@10:\t{:4f}".format(p_at_1, p_at_5, p_at_10))
        num_candidates = rep_trg.shape[0]


        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, p_at_1, p_at_5, p_at_10, num_candidates])
            else:
                with open(csv_path, mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, p_at_1, p_at_5, p_at_10, num_candidates])



        return p_at_1

def compute_PA(x, y):
    if True:
        print('a')
        print(x.shape)
        print(y.shape)
        # x = torch.transpose(x, 0, 1)
        # y = torch.transpose(y, 0, 1)
        print('b')
        print(x.shape)
        print(y.shape)
        U, S, Vt = torch.svd(torch.mm(torch.transpose(y, 0, 1), x))
        W = torch.mm(U, Vt)
        print(W.shape)
        # apply mappping
        x_mapped = torch.transpose(torch.mm(W, torch.transpose(x, 0, 1)), 0, 1)
        print(x_mapped.shape)
        print(x_mapped - y)
        return W

if __name__=="__main__":
    _x = torch.tensor(np.randint((30, 10)))
    y = torch.tensor(np.randint((30, 10)))
    m = torch.tensor(np.randint((10, 10)))
    x = torch.mm(_x, m)

    compute_PA(x, y)


