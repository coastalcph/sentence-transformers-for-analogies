from . import InputExample
import gzip
import os
from analogy.data import read_analogy_data, is_comment


class BiDictReader(object):
    """
    Reads in a bilingual dict
    """
    def __init__(self):
        pass

    def get_examples(self, filename, filename_candidates=None, max_examples=0, sep='\t'):
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
        if filename_candidates:
            for line in open(filename_candidates):
                trg_word = line.strip()
                if trg_word not in seen_trg:
                    seen_trg.add(trg_word)
                    trg2id[trg_word] = len(trg_words)
                    trg_words.append(InputExample(guid='', texts=[trg_word], label=1))
        return src_words, trg_words, src2trg
