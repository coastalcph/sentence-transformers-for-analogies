from torch.utils.data import Dataset
from typing import List
import torch
import logging
from tqdm import tqdm
from .. import SentenceTransformer
from ..readers.InputExample import InputExample


class BDIDataset(Dataset):
    """

    """
    def __init__(self, examples: List[InputExample], model: SentenceTransformer, show_progress_bar: bool = None):
        if show_progress_bar is None:
            show_progress_bar = (logging.getLogger().getEffectiveLevel() == logging.INFO or logging.getLogger().getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar
        self.convert_input_examples(examples, model)

    def convert_input_examples(self, examples: List[InputExample], model: SentenceTransformer):
        """
        Converts input examples to a SmartBatchingDataset usable to train the model with
        SentenceTransformer.smart_batching_collate as the collate_fn for the DataLoader

        smart_batching_collate as collate_fn is required because it transforms the tokenized texts to the tensors.

        :param examples:
            the input examples for the training
        :param model
            the Sentence BERT model for the conversion
        :return: a SmartBatchingDataset usable to train the model with SentenceTransformer.smart_batching_collate as the collate_fn
            for the DataLoader
        """

        inputs = []
        too_long = 0

        iterator = examples
        max_seq_length = model.get_max_seq_length()

        if self.show_progress_bar:
            iterator = tqdm(iterator, desc="Encode dataset")

        for ex_index, example in enumerate(iterator):

            tokenized_text = model.tokenize(example.texts[0])

            if max_seq_length != None and max_seq_length > 0 and len(tokenized_text) >= max_seq_length:
                too_long += 1


                #inputs[i].append(tokenized_texts[i])
            inputs.append(tokenized_text)

        logging.info("Num examples: %d" % (len(examples)))
        logging.info("Examples longer than max_seqence_length: {}".format(too_long))

        self.tokens = inputs

    def __getitem__(self, item):
        return [self.tokens[item], None]

    def __len__(self):
        return len(self.tokens)