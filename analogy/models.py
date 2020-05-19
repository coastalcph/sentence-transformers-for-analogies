import logging
import copy

import torch
from torch import nn

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
    def __init__(self, train_embeddings, test_embeddings, other_embeddings, reg_term_lambda=0.001, delta=0.1):
        super(AnalogyModel, self).__init__()
        self.train_embeddings = train_embeddings
        self.original_embeddings = copy.deepcopy(train_embeddings)
        self.test_embeddings = test_embeddings
        self.other_embeddings = other_embeddings
        self.loss = nn.CosineEmbeddingLoss()
        self.regularization = nn.MSELoss(reduction='sum')
        self.reg_term_lambda = reg_term_lambda
        self.delta = delta

    def set_mapper(self, mapper):
        self.mapper = mapper

    def loss_function(self, x, y):
        if self.training:
            e1 = x['e1']
            e2 = x['e2']
            e3 = x['e3']
            e4 = x['e4']
            offset_trick = x['offset_trick']
            scores = x['scores']
            distances = x['distances']
            t_l = x['t_l']
            t_r = x['t_r']
            batch_size = e1.shape[0]
            e3_embeddings = self.train_embeddings(e3)
            entities = torch.cat([e1, e2, e3, e4]).unique()
            reg_term = self.regularization(self.original_embeddings(entities), self.train_embeddings(entities))
            score = torch.bmm(offset_trick.view(batch_size, 1, -1), e3_embeddings.view(batch_size, -1, 1)).squeeze()
            neg_left_score = torch.bmm(offset_trick.view(batch_size, 1, -1), t_l.view(batch_size, -1, 1)).squeeze()
            neg_right_score = torch.bmm(e3_embeddings.view(batch_size, 1, -1), t_r.view(batch_size, -1, 1)).squeeze()
            left_loss = nn.functional.relu(self.delta + neg_left_score - score).sum()
            right_loss = nn.functional.relu(self.delta + neg_right_score - score).sum()
            loss = left_loss + right_loss + self.reg_term_lambda * reg_term
            return loss
            # return self.loss(offset_trick, self.trainable_embeddings(e3), y) + self.reg_term_lambda * reg_term
        else:
            e3 = x['e3']
            offset_trick = x['offset_trick']
            return self.loss(offset_trick, self.mapper.apply(self.test_embeddings(e3)), y)

    def is_success(self, e3, e1_e2_e4, top4):
        if e3 not in top4:
            return False
        else:
            for elem in top4:
                if elem != e3 and elem not in e1_e2_e4:
                    return False
                if elem == e3:
                    return True

    def accuracy(self, x, y):
        e1s = x['e1']
        e2s = x['e2']
        e3s = x['e3']
        e4s = x['e4']
        scores = x['scores']
        sorted_indexes_by_scores = scores.argsort(descending=True)[:, :4]
        accuracies = list()
        for e1, e2, e3, e4, top4_indexes in zip(e1s, e2s, e3s, e4s, sorted_indexes_by_scores):
            success = self.is_success(e3, {e1, e2, e4}, top4_indexes)
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

        if self.training:
            e1_embeddings = self.train_embeddings(e1)
            e2_embeddings = self.train_embeddings(e2)
            e3_embeddings = self.train_embeddings(e3)
            e4_embeddings = self.train_embeddings(e4)
            offset_trick = e1_embeddings - e2_embeddings + e4_embeddings
            a_norm = offset_trick / offset_trick.norm(dim=1)[:, None]
            t_l = a_norm[a_norm.matmul(a_norm.transpose(0, 1)).argsort()[:, -2]]
            e3_norm = e3_embeddings / e3_embeddings.norm(dim=1)[:, None]
            t_r = e3_norm[e3_norm.matmul(e3_norm.transpose(0, 1)).argsort()[:, -2]]
            b_norm = self.train_embeddings.weight / self.train_embeddings.weight.norm(dim=1)[:, None]
            cosine_sims = torch.mm(a_norm, b_norm.transpose(0,1))
            return {
                "e1": e1,
                "e2": e2,
                "e3": e3,
                "e4": e4,
                "offset_trick": offset_trick,
                "scores": cosine_sims,
                "distances": distances,
                "t_l": t_l,
                "t_r": t_r,

            }
        else:
            e1_embeddings = self.test_embeddings(e1)
            e2_embeddings = self.test_embeddings(e2)
            e3_embeddings = self.test_embeddings(e3)
            e4_embeddings = self.test_embeddings(e4)
            mapped_e1 = self.mapper.apply(e1_embeddings)
            mapped_e2 = self.mapper.apply(e2_embeddings)
            mapped_e3 = self.mapper.apply(e3_embeddings)
            mapped_e4 = self.mapper.apply(e4_embeddings)
            offset_trick = mapped_e1 - mapped_e2 + mapped_e4 
            a_norm = offset_trick / offset_trick.norm(dim=1)[:, None]
            mapped_embedding_table = self.mapper.apply(self.test_embeddings.weight)
            b_norm = mapped_embedding_table / mapped_embedding_table.norm(dim=1)[:, None]
            cosine_sims = torch.mm(a_norm, b_norm.transpose(0,1))
            return {
                "e1": e1,
                "e2": e2,
                "e3": e3,
                "e4": e4,
                "offset_trick": offset_trick,
                "scores": cosine_sims,
                "distances": distances,
            }



class IdentityMapper:
    def apply(self, elems):
        return elems


class NeuralMapper:
    def __init__(self, mapping_model, device):
        self.model = mapping_model
        self.device = device

    def apply(self, elems):
        return self.model(elems)
