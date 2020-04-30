from . import InputExample
import gzip
import os
from analogy.data import read_analogy_data, is_comment


class AnalogyReader(object):
    """
    Reads in the Stanford NLI dataset and the MultiGenre NLI dataset
    """
    def __init__(self):
        pass

    def get_examples(self, filename, max_examples=0):
        """
        """
        examples = []
        id = 0
        for row in read_analogy_data(filename):
            if not is_comment(row):
                guid = "%s-%d" % (filename, id)
                id += 1

                examples.append(InputExample(guid=guid, texts=[row['Q1'], row['Q2'], row['Q3'], row['Q4']], label=1))

                if 0 < max_examples <= len(examples):
                    break

        return examples

    """
    @staticmethod
    def get_labels():
        return {"contradiction": 0, "entailment": 1, "neutral": 2}

    def get_num_labels(self):
        return len(self.get_labels())

    def map_label(self, label):
        return self.get_labels()[label.strip().lower()]
    """