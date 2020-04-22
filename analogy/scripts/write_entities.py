import logging
import os

from data import build_analogy_examples_from_file


def get_entities(fname):
    logging.info("Writing entities for file: {}".format(fname))
    analogies = build_analogy_examples_from_file(fname)
    entities = set()
    for analogy in analogies:
        entities.add(analogy.q_1_source)
        entities.add(analogy.q_1_target)
        entities.add(analogy.q_2_source)
        entities.add(analogy.q_2_target)
    return entities



def main():
    data_dir = './data/analogy_qids/'
    files = os.listdir(data_dir)
    files = [f for f in files if 'all' in f]
    for f in files:
        entities = get_entities(os.path.join(data_dir, f))
        with open('./data/words/{}.words.txt'.format(f), 'w') as fhandle:
            for e in entities:
                fhandle.write("{}\n".format(e.lower()))


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
