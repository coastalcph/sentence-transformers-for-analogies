import logging
from dataclasses import dataclass
import csv


@dataclass
class Analogy:
    q_1_type: str
    q_1_id: str
    q_2_type: str
    q_2_id: str
    p_type: str
    p_id: str
    q_1_source: str
    q_1_source_id: str
    q_1_target: str
    q_1_target_id: str
    q_2_source: str
    q_2_source_id: str
    q_2_target: str
    q_2_target_id: str

    def __repr__(self):
        return """
        ====================
    Analogy: {q_1_type} ({q_1_id}) -> {p_type} ({p_id}) -> {q_2_type} ({q_2_id})
    ====================
    {q_1_source} ({q_1_source_id}) -> {q_1_target} ({q_1_target_id}) / {q_2_source} ({q_2_source_id}) -> {q_2_target} ({q_2_target_id})
    ____________________
        """.format(
            q_1_type=self.q_1_type,
            q_1_id=self.q_1_id,
            p_type=self.p_type,
            p_id=self.p_id,
            q_2_type=self.q_2_type,
            q_2_id=self.q_2_id,
            q_1_source=self.q_1_source,
            q_1_source_id=self.q_1_source_id,
            q_1_target=self.q_1_target,
            q_1_target_id=self.q_1_target_id,
            q_2_source=self.q_2_source,
            q_2_source_id=self.q_2_source_id,
            q_2_target=self.q_2_target,
            q_2_target_id=self.q_2_target_id,
        )




def read_analogy_data(fname):
    logging.info("Loading analogy file: {}".format(fname))
    with open(fname, newline='') as csvfile:
        fieldnames = ['Q1', 'Q1_id', 'Q2', 'Q2_id', 'Q3', 'Q3_id', 'Q4', 'Q4_id']
        reader = csv.DictReader(csvfile, delimiter=';', fieldnames=fieldnames)
        for row in reader:
            yield row


def is_comment(row):
    return row['Q1'].startswith('# ')


def parse_type_and_id(row):
    # Parsing Q1
    q1, other = row['Q1'].split('>--')
    q1 = q1[2:].strip()
    q1_elems = q1.split('(')
    q1_id = q1_elems[-1]
    q1_type = " ".join(q1_elems[:-1])
    q1_type = q1_type.strip()
    q1_id, _ = q1_id.split(')')

    # Parsing P
    p, q2 = other.split('-->')
    p_elems = p.split('(')
    p_id = p_elems[-1]
    p_type = " ".join(p_elems[:-1])
    p_type = p_type.strip()
    p_id, _ = p_id.split(')')

    # Parsing Q2
    q2_elems = q2.split('(')
    q2_id = q2_elems[-1]
    q2_type = " ".join(q2_elems[:-1])
    q2_type = q2_type.strip()
    q2_id, _ = q2_id.split(')')

    return q1_type, q1_id, p_type, p_id, q2_type, q2_id


def build_analogy_examples(rows):
    logging.info("Building analogy examples")
    current_q_1_type = None
    current_q_1_id = None
    current_q2_type = None
    current_q2_id = None
    current_p_type = None
    current_p_id = None

    for row in rows:
        if is_comment(row):
            types_and_ids = parse_type_and_id(row)
            (current_q_1_type,
             current_q_1_id,
             current_p_type,
             current_p_id,
             current_q2_type,
             current_q2_id) = types_and_ids
        else:
            yield Analogy(
                current_q_1_type,
                current_q_1_id,
                current_p_type,
                current_p_id,
                current_q2_type,
                current_q2_id,
                row['Q1'],
                row['Q1_id'],
                row['Q2'],
                row['Q2_id'],
                row['Q3'],
                row['Q3_id'],
                row['Q4'],
                row['Q4_id'],
            )


def build_analogy_examples_from_file(fname):
    return build_analogy_examples(read_analogy_data(fname))


if __name__ == '__main__':
    analogies = build_analogy_examples(read_analogy_data('./data/analogy_qids/analogy_unique_da.csv'))
    print(list(analogies))

