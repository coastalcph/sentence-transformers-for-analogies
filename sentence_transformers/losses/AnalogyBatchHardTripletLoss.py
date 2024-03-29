from typing import Iterable, Dict

import torch
import numpy as np
from torch import nn, Tensor

#from ..util import combine_anchor_entities


class AnalogyBatchHardTripletLoss(nn.Module):
    def __init__(self, sentence_embedder, triplet_margin: float = 1):
        super(AnalogyBatchHardTripletLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sentence_embedder = sentence_embedder
        self.triplet_margin = triplet_margin


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.sentence_embedder(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]

        ea, e3 = self.combine_anchor_entities(reps[0], reps[1], reps[2], reps[3])
        embeddings = torch.cat((ea, e3), 0)
        # labels should be the same only for *corresponding* anchor and rep_e3
        _labels = torch.from_numpy(np.array([i for i in range(ea.shape[0])]))
        labels = torch.cat((_labels, _labels), 0).to(self.device)
        return AnalogyBatchHardTripletLoss.batch_hard_triplet_loss(labels, embeddings, margin=self.triplet_margin)

    def combine_anchor_entities(self, e1, e2, e3, e4):
        return e1 - e2 + e4, e3

    # Hard Triplet Loss for Analogies
    # Adapted from sentenceBert where it was adapted from:
    # Source: https://github.com/NegatioN/OnlineMiningTripletLoss/blob/master/triplet_loss.py
    # Paper: In Defense of the Triplet Loss for Person Re-Identification, https://arxiv.org/abs/1703.07737
    @staticmethod
    def batch_hard_triplet_loss(labels: Tensor, embeddings: Tensor, margin: float, squared: bool = False) -> Tensor:
        """Build the triplet loss over a batch of embeddings.
        For analogies, there is usually only one positive, which is therefor the hardest positive
        not every embedding can serve as an anchor (only (e1-e2+e4) should be anchors)
        For each anchor, we get the hardest negative to form a triplet.
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            Label_Sentence_Triplet: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        pairwise_dist = AnalogyBatchHardTripletLoss._pairwise_distances(embeddings, squared=squared)

        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same label)
        # for the analogies, there is only one positive triplet per anchor, which is indicated in the labels
        mask_anchor_positive = AnalogyBatchHardTripletLoss._get_anchor_positive_triplet_mask(labels).float()

        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = mask_anchor_positive * pairwise_dist

        # shape (batch_size, 1)
        hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels and are a pair of an analogy anchor (e1-e2+e4) and an e3 repr)
        mask_anchor_negative = AnalogyBatchHardTripletLoss._get_anchor_negative_triplet_mask(labels).float()

        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

        # shape (batch_size,)
        hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        tl = hardest_positive_dist - hardest_negative_dist + margin


        # select only the losses for the valid anchors (as e3 cannot be an anchor) , i.e. the first bs elements
        bs = int(embeddings.shape[0]/2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss_mask = torch.from_numpy(np.array([1]*bs + [0]*bs).astype(np.float32)).reshape(tl.shape).to(device)

        tl = tl*loss_mask

        tl[tl < 0] = 0
        triplet_loss = tl.sum()/bs

        return triplet_loss

    def batch_all_triplet_loss(self, labels, embeddings, margin, squared=False):
        """Build the triplet loss over a batch of embeddings.
        We generate all the valid triplets and average the loss over the positive ones.
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            Label_Sentence_Triplet: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        pairwise_dist = self._pairwise_distances(embeddings, squared=squared)

        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        mask = self._get_triplet_mask(labels)
        triplet_loss = mask.float() * triplet_loss

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss[triplet_loss < 0] = 0

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = triplet_loss[triplet_loss > 1e-16]
        num_positive_triplets = valid_triplets.size(0)
        num_valid_triplets = mask.sum()

        fraction_positive_triplets = num_positive_triplets / (num_valid_triplets.float() + 1e-16)

        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

        return triplet_loss, fraction_positive_triplets

    @staticmethod
    def _pairwise_distances(embeddings, squared=False):
        """Compute the 2D matrix of distances between all the embeddings.
        Args:
            embeddings: tensor of shape (batch_size, embed_dim)
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            pairwise_distances: tensor of shape (batch_size, batch_size)
        """
        dot_product = torch.matmul(embeddings, embeddings.t())

        # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
        # This also provides more numerical stability (the diagonal of the result will be exactly 0).
        # shape (batch_size,)
        square_norm = torch.diag(dot_product)

        # Compute the pairwise distance matrix as we have:
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)

        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances[distances < 0] = 0

        if not squared:
            # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
            # we need to add a small epsilon where distances == 0.0
            mask = distances.eq(0).float()
            distances = distances + mask * 1e-16

            distances = (1.0 - mask) * torch.sqrt(distances)

        return distances

    @staticmethod
    def _get_triplet_mask(labels):
        """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
        A triplet (i, j, k) is valid if:
            - i, j, k are distinct
            - labels[i] == labels[j] and labels[i] != labels[k]
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        """
        # Check that i, j and k are distinct
        indices_equal = torch.eye(labels.size(0)).byte()
        indices_not_equal = ~indices_equal
        i_not_equal_j = indices_not_equal.unsqueeze(2)
        i_not_equal_k = indices_not_equal.unsqueeze(1)
        j_not_equal_k = indices_not_equal.unsqueeze(0)

        distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

        label_equal = labels.unsqueeze(0)==labels.unsqueeze(1)
        i_equal_j = label_equal.unsqueeze(2)
        i_equal_k = label_equal.unsqueeze(1)

        valid_labels = ~i_equal_k & i_equal_j

        return valid_labels & distinct_indices

    @staticmethod
    def _get_anchor_positive_triplet_mask(labels):
        """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Check that i and j are distinct

        device = 'cuda' if labels.is_cuda else 'cpu'
        indices_equal = torch.eye(labels.size(0)).byte().to(device)
        indices_not_equal = ~indices_equal
        # Check if labels[i] == labels[j]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        labels_equal = labels.unsqueeze(0)==labels.unsqueeze(1)
        return labels_equal & indices_not_equal

    @staticmethod
    def _get_anchor_negative_triplet_mask(labels):
        """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels AND are a pair of (e1-e2+e4) and e3
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Check if labels[i] != labels[k]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        unequal_labels = ~(labels.unsqueeze(0)==labels.unsqueeze(1))
        # select only the bs last columns
        bs = int(labels.shape[0]/2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        anchor_e3_pairs = torch.from_numpy(np.hstack([np.zeros((bs*2, bs)), np.ones((bs*2, bs))]).astype(np.uint8)).to(device)
        return unequal_labels & anchor_e3_pairs

if __name__=="__main__":
    import numpy as np
    num_data = 5
    num_classes = num_data
    emb_dim = 10
    anchors = torch.from_numpy(np.random.randint(0, 100, size=(num_data, emb_dim)).astype(np.float32))
    e3s = torch.from_numpy(np.random.randint(0, 100, size=(num_data, emb_dim)).astype(np.float32))
    e4s = torch.from_numpy(np.random.randint(0, 100, size=(num_data, emb_dim)).astype(np.float32))
    sims = torch.mm(e4s, e3s.transpose(0,1))
    """
    print(anchors.shape[0])
    _labels = torch.from_numpy(np.array([i for i in range(anchors.shape[0])]))
    labels = torch.cat((_labels, _labels), 0)
    print(labels)

    embeddings = torch.cat((anchors, e4s), 0)

    l = AnalogyBatchHardTripletLoss
    mp = l._get_anchor_positive_triplet_mask(labels)
    mn = l._get_anchor_negative_triplet_mask(labels)
    print(mn)
    ls = l.batch_hard_triplet_loss(labels=labels, embeddings= embeddings, margin=1)
    print(ls)
    """
    print(sims)
    idxs = sims.argsort(descending=True)[:, 0]
    e3_idxs = torch.from_numpy(np.array([elm for elm in range(num_data)]).astype(np.long))
    print(idxs)
    print(e3_idxs)
    print(len(e3_idxs))
    print(idxs - e3_idxs)

