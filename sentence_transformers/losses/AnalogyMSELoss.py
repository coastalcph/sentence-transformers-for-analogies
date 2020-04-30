import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
from ..SentenceTransformer import SentenceTransformer
import logging

class AnalogyMSELoss(nn.Module):
    """
    optimizes MSE loss between (e1-e2+e4) and e3
    """
    def __init__(self,
                 model: SentenceTransformer):
        super(AnalogyMSELoss, self).__init__()
        self.model = model



    def combine_anchor_entities(self, e1, e2, e4):
        return e1 - e2 + e4


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        rep_e1, rep_e2, rep_e3, rep_e4 = reps
        anchor = self.combine_anchor_entities(rep_e1, rep_e2, rep_e4)

        loss_fct = nn.MSELoss()
        loss = loss_fct(anchor, rep_e3)
        return loss

