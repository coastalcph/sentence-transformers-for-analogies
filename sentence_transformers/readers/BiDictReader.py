from . import InputExample
import numpy as np

class BiDictReader(object):
    """
    Reads in a bilingual dict
    """
    def __init__(self):
        pass

    def get_examples(self, filename, filename_candidates=None, max_examples=0, sep='\t', num_candidates=200000):
        """
        """
        src_words = []
        trg_words = []
        id = 0
        src2trg = {}
        src2id = {}
        trg2id = {}
        seen_src = set()
        seen_trg = set()
        for line in open(filename):
            if id == 0:
                # check if separator is correct
                if len(line.split(sep)) != 2:
                    if sep == '\t': sep = ' '
                    elif sep == ' ': sep = '\t'
            guid = "%s-%d" % (filename, id)
            id += 1
            src_word = line.strip().split(sep)[0]
            trg_word = line.strip().split(sep)[1]
            # filter out pairs consisting of identical strings
            if src_word != trg_word:
                if not src_word in seen_src:
                    src2id[src_word] = len(src_words)
                    src_words.append(InputExample(guid=guid, texts=[src_word], label=1))
                    seen_src.add(src_word)
                if not trg_word in seen_trg:
                    trg2id[trg_word] = len(trg_words)
                    trg_words.append(InputExample(guid=guid, texts=[trg_word], label=1))
                    seen_trg.add(trg_word)
                src2trg.setdefault(src2id[src_word], []).append(trg2id[trg_word])
                if 0 < max_examples <= len(src_words):
                    break
        # add additional candidates
        minimum_trg_words = len(trg_words)
        if filename_candidates:

            i = 0
            for line in open(filename_candidates):
                if i == 0:
                    i += 1
                    continue
                i+= 1
                trg_word = line.split(' ')[0]

                if trg_word not in seen_trg:
                    seen_trg.add(trg_word)
                    trg2id[trg_word] = len(trg_words)
                    trg_words.append(InputExample(guid='', texts=[trg_word], label=1))
        trg_words = trg_words[:int(np.max([num_candidates, minimum_trg_words]))]
        print('Size of candidate set: {}'.format(len(trg_words)))
        return src_words, trg_words, src2trg
