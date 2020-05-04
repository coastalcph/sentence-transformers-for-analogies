from . import SentenceEvaluator, SimilarityFunction
from torch.utils.data import DataLoader
import os
import torch
import logging
from tqdm import tqdm
from ..util import batch_to_device, combine_anchor_entities
import csv
from transformers import BertModel, BertTokenizer

import numpy as np

class AnalogyEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on its ability to complete analogies.
    For each analogy, compute the closest entity to (e1-e2+e4) and check if its e3

    Returned score is accuracy over the whole dataset

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """


    def __init__(self, dataloader: DataLoader, main_similarity: SimilarityFunction = None, name: str = '', show_progress_bar: bool = None, write_predictions: bool=False, tokenizer: BertTokenizer=None):
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
        self.csv_headers = ["epoch", "steps", "accuracy"]
        self.write_predictions=write_predictions
        self.tokenizer = tokenizer

    """
    def __call__(self, model: 'SequentialSentenceEmbedder', output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.eval()
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logging.info("Evaluation the model on "+self.name+" dataset"+out_txt)

        self.dataloader.collate_fn = model.smart_batching_collate

        iterator = self.dataloader
        if self.show_progress_bar:
            iterator = tqdm(iterator, desc="Convert Evaluating")

        rep_ea = []
        rep_e3 = []
        analogies = []
        # encode all analogies

        for step, batch in enumerate(iterator):
            features, label_ids = batch_to_device(batch, self.device)
            with torch.no_grad():
                reps = [model(sent_features)['sentence_embedding'] for sent_features in features]

                ea, e3 = combine_anchor_entities(reps[0], reps[1], reps[2], reps[3])
                # candidates are all single entities in the dataset

                bs = ea.shape[0]
                rep_ea.append(ea)
                rep_candidates.append(e3)

                if self.write_predictions:
                    # de-tokenize
                    analogy_batch =  [[] for x in range(bs)]
                    for eid, sent_features in enumerate(features):
                        input_ids = sent_features['input_ids']
                        # iterate through batch
                        for i, input in enumerate(input_ids):
                            assert len(input) > 0
                            surface = self.tokenizer.convert_ids_to_tokens(input)
                            analogy_batch[i].append(' '.join(surface))

                    analogies.extend(analogy_batch)

        rep_ea = torch.cat(rep_ea, 0)
        rep_e3 = torch.cat(rep_e3, 0)

        ################################
        ####### Compute cosines sims ####
        #################################
        num_data = rep_ea.shape[0]
        a_norm = rep_ea / rep_ea.norm(dim=1)[:, None]
        b_norm = rep_e3 / rep_e3.norm(dim=1)[:, None]
        cosine_sims = torch.mm(a_norm, b_norm.transpose(0, 1))

        top_ten_idxs = cosine_sims.argsort(descending=True)[:,:10]

        retrieved_idxs = top_ten_idxs[:,0]
        correct_idxs = torch.from_numpy(np.array([elm for elm in range(num_data)]).astype(np.long)).to(self.device)
        accuracy = ((retrieved_idxs - correct_idxs) == 0).float().sum() / num_data
        logging.info("Accuracy:\t{:4f}".format(accuracy))

        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, accuracy.item()])
            else:
                with open(csv_path, mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, accuracy.item()])


        if self.write_predictions:

            assert len(analogies) == cosine_sims.shape[0]
            with open(os.path.join(output_path, 'predictions_{}.csv'.format(0)), 'w') as fpred:
                pred_writer = csv.writer(fpred, delimiter=';')
                # write out the model predictions
                top_ten = top_ten_idxs.cpu().numpy()
                for aid in range(cosine_sims.shape[0]):
                    predictions = [analogies[pid][2] for pid in top_ten[aid]]
                    pred_writer.writerow([analogies[aid][0], analogies[aid][1], analogies[aid][2], analogies[aid][3],  ','.join(predictions)])
            fpred.close()
        return accuracy

    """


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
                        surface = self.tokenizer.convert_ids_to_tokens(input)
                        analogy_batch[i].append(' '.join(surface))
                # for each analogy, add all entities to the candidate set if not present yet
                for es, rep in zip(analogy_batch, reps):
                    e1, e2, e3, e4 = es[0], es[1], es[2], es[3]

                    if e1 not in candidate2id:
                        candidate2id[e1] = len(rep_candidates)
                        id2candidate[candidate2id[e1]] = e1
                        rep_candidates.append(reps[0])
                    if e2 not in candidate2id:
                        candidate2id[e2] = len(rep_candidates)
                        rep_candidates.append(reps[1])
                        id2candidate[candidate2id[e2]] = e2
                    if e3 not in candidate2id:
                        candidate2id[e3] = len(rep_candidates)
                        rep_candidates.append(reps[2])
                        id2candidate[candidate2id[e3]] = e3
                    if e4 not in candidate2id:
                        candidate2id[e4] = len(rep_candidates)
                        rep_candidates.append(reps[3])
                        id2candidate[candidate2id[e4]] = e4
                    analogy_ids.append([candidate2id[e1], candidate2id[e2], candidate2id[e3], candidate2id[e4]])

                rep_ea.append(ea)
                analogies.extend(analogy_batch)
                analogies2ids.extend(analogy_ids)

        rep_ea = torch.cat(rep_ea, 0)
        rep_candidates = torch.cat(rep_candidates, 0)
        assert rep_candidates.shape[0] == len(id2candidate) == len(candidate2id)
        print(rep_ea.shape)
        print(rep_candidates.shape)
        print(len(analogies2ids))
        ################################
        ####### Compute cosines sims ####
        #################################
        num_data = rep_ea.shape[0]
        a_norm = rep_ea / rep_ea.norm(dim=1)[:, None]
        b_norm = rep_candidates / rep_candidates.norm(dim=1)[:, None]
        cosine_sims = torch.mm(a_norm, b_norm.transpose(0, 1))

        top_ten_idxs = cosine_sims.argsort(descending=True)[:, :10]

        retrieved_idxs = top_ten_idxs[:, 0]
        correct_idxs = torch.from_numpy(np.array([elm[2] for elm in analogies2ids]).astype(np.long)).to(self.device)
        accuracy = ((retrieved_idxs - correct_idxs) == 0).float().sum() / num_data
        logging.info("Accuracy:\t{:4f}".format(accuracy))

        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, accuracy.item()])
            else:
                with open(csv_path, mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, accuracy.item()])

        if self.write_predictions:

            assert len(analogies) == cosine_sims.shape[0]
            with open(os.path.join(output_path, 'predictions_{}.csv'.format(0)), 'w') as fpred:
                pred_writer = csv.writer(fpred, delimiter=';')
                # write out the model predictions
                top_ten = top_ten_idxs.cpu().numpy()
                for aid in range(cosine_sims.shape[0]):
                    predictions = [id2candidate[pid] for pid in top_ten[aid]]
                    pred_writer.writerow([analogies[aid][0], analogies[aid][1], analogies[aid][2], analogies[aid][3],
                                          ','.join(predictions)])
            fpred.close()
        return accuracy

