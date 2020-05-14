from torch import nn, Tensor
from typing import Iterable, Dict
from ..SentenceTransformer import SentenceTransformer
from sentence_transformers.util import combine_anchor_entities


class AnalogyCosineLoss(nn.Module):
    """
    optimizes MSE loss between (e1-e2+e4) and e3
    """
    def __init__(self,
                 model: SentenceTransformer):
        super(AnalogyCosineLoss, self).__init__()
        self.model = model


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        ea, e3 = combine_anchor_entities(reps[0], reps[1], reps[2], reps[3])
        loss_fct = nn.CosineEmbeddingLoss()
        loss = loss_fct(ea, e3)
        return loss

