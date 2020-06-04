from . import InputExample
import gzip
import os
from analogy.data import read_analogy_data, is_comment


class AnalogyReader(object):
    """
    Reads in the Wiking dataset
    """
    def __init__(self, use_context):
        self.use_context = True if use_context == 1 else False

    def format_text_with_context(self, title, context, filters=None):
        new_title = title
        if self.use_context and context != '':
            context = eval(context)
            aliases = context['aliases']
            description = context['description']

            if aliases:
                filtered_aliases = [a for a in aliases if a.lower() != title.lower() or a.lower() in title.lower()]
                aliases_txt = ", ".join(filtered_aliases)[:100]

                if filters is not None:
                    for f in filters:
                        elems = f.split()
                        for elem_to_remove in elems:
                            aliases_txt = aliases_txt.replace(elem_to_remove, "[UNK]")

                new_title += " [SEP] {}".format(aliases_txt)

            if description:
                if filters is not None:
                    for f in filters:
                        elems = f.split()
                        for elem_to_remove in elems:
                            description = description.replace(elem_to_remove, "[UNK]")
                new_title += " [SEP] {}".format(description[:100])

        return new_title


    def get_examples(self, filename, max_examples=0):
        """
        """
        examples = []
        id = 0
        for row in read_analogy_data(filename):
            if not is_comment(row):
                try:
                    guid = "%s-%d" % (filename, id)
                    text_1 = self.format_text_with_context(row['Q1'], row['Q1_context'], [row['Q3']])
                    text_2 = self.format_text_with_context(row['Q2'], row['Q2_context'], [row['Q3']])
                    text_3 = self.format_text_with_context(row['Q3'], row['Q3_context'], [row['Q1'], row['Q2'], row['Q4']])
                    text_4 = self.format_text_with_context(row['Q4'], row['Q4_context'], [row['Q3']])
                    examples.append(InputExample(guid=guid, texts=[text_1, text_2, text_3, text_4], label=1))
                    id += 1
                except Exception as e:
                    import pdb; pdb.set_trace()
                    print("Could not parse row: {}".format(row))
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

