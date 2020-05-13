import numpy as np
from scipy.stats.stats import pearsonr
from poutyne.framework.metrics import EpochMetric


class CorrelationMetric(EpochMetric):
    def __init__(self) -> None:
        super().__init__()
        self.scores = list()
        self.distances = list()

    def forward(self, x, y):
        # Accumulate metrics here
        e3 = x['e3']
        scores = x['scores']
        distances = x['distances']
        for i, (s, d) in enumerate(zip(scores, distances)):
            if d >= 0:
                self.scores.append(1 - float(s[e3[i]]))
                self.distances.append(float(d))  # We append the distance

    def get_metric(self):
        val =  np.corrcoef(self.scores, self.distances)[0][1]
        self.reset()
        return val

    def reset(self) -> None:
        self.scores = list()
        self.distances = list()


class CorrelationBinnedAccuracyMetric(EpochMetric):
    def __init__(self) -> None:
        super().__init__()
        self.accuracies = list()
        self.distances = list()

    def is_success(self, e3, e1_e2_e4, top4):
        if e3 not in top4:
            return False
        else:
            for elem in top4:
                if elem != e3 and elem not in e1_e2_e4:
                    return False
                if elem == e3:
                    return True

    def store_accuracy(self, x, y):
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
                self.accuracies.append(1)
            else:
                self.accuracies.append(0)

    def forward(self, x, y):
        # Accumulate metrics here
        self.store_accuracy(x, y)
        self.distances += x['distances']

    def get_metric(self):
        zero_three = list()
        three_four = list()
        four_five = list()
        five_six = list()
        six_rest = list()
        for a, d in zip(self.accuracies, self.distances):
            if d < 0.3 and d >=0 :
                zero_three.append(a)
            elif d < 0.4:
                three_four.append(a)
            elif d < 0.5:
                four_five.append(a)
            elif d < 0.6:
                five_six.append(a)
            elif d >= 0.6:
                six_rest.append(a)
        print()
        print("Correlation bins")
        print("="*80)
        print(len(self.accuracies), len(self.distances))
        print("{}\t{}/{}\t{}".format("0.0-0.3", sum(zero_three), len(zero_three), sum(zero_three) / len(zero_three)))
        print("{}\t{}/{}\t{}".format("3.0-0.4", sum(three_four), len(three_four), sum(three_four) / len(three_four)))
        print("{}\t{}/{}\t{}".format("4.0-0.5", sum(four_five), len(four_five), sum(four_five) / len(four_five)))
        print("{}\t{}/{}\t{}".format("5.0-0.6", sum(five_six), len(five_six), sum(five_six) / len(five_six)))
        print("{}\t{}/{}\t{}".format("6.0-1.0", sum(six_rest), len(six_rest), sum(six_rest) / len(six_rest)))
        print("="*80)
        print()
        self.reset()
        return 0.0

    def reset(self) -> None:
        self.accuracies = list()
        self.distances = list()
