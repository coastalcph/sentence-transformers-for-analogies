import numpy as np
from poutyne.framework.metrics import EpochMetric


class CorrelationMetric(EpochMetric):
    def __init__(self) -> None:
        super().__init__()
        self.scores = list()
        self.distances = list()

    def forward(self, x, y):
        # Accumulate metrics here
        e1, e2, e3, e4, offset_trick, scores, distances = x
        for i, (s, d) in enumerate(zip(scores, distances)):
            self.scores.append(1 - float(s[e3[i]]))
            self.distances.append(float(d))  # We append the distance

    def get_metric(self):
        return np.corrcoef(self.scores, self.distances)[0][1]

    def reset(self) -> None:
        self.scores = list()
        self.distances = list()
