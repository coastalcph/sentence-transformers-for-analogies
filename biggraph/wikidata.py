import gzip
import json
import csv
from analogy.data import read_analogy_data, is_comment
import configparser
from biggraph.graphreader import get_q_json, load_pointers, file_open
import os
'''
methods for parsing wikidata json dumps
'''

DESCRIPTION='descriptions'
ALIAS='aliases'
LABEL='labels'
SUBCAT='subcat'
SUMMARY='summary'
LABELONE='labelone'



def gz_reader(fname):
    with gzip.open(fname, 'rb') as f:
        for line in f:
            if len(line.decode().strip().strip(',')) > 1:
                yield json.loads(line.decode().strip().strip(','))


def json_dump_reader(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.strip().strip(',')) > 1:
                yield json.loads(line.strip().strip(','))


def get_description(j, lang):
    if lang in j['descriptions']:
        return j['descriptions'][lang]['value']
    else:
        return None


def get_aliases(j, lang):
    if lang in j['aliases']:
        return [elm['value'] for elm in j['aliases'][lang]]
    else:
        return None


def get_label(j, lang):
    if lang in j['labels']:
        return j['labels'][lang]['value']
    else:
        return None


def get_qid(j):
    return j['id']

def get_properties(j):
    """
    returns a list of property-qcode pairs
    :param j:
    :return:
    """
    property_tuples = []
    if 'claims' in j.keys():
        for p, r in j['claims'].items():
            for elm in r:
                if has_key(elm, 'mainsnak'):
                    elm = elm['mainsnak']
                    if has_key(elm, 'datavalue'):
                        if has_key(elm['datavalue'], 'value'):
                            if has_key(elm['datavalue']['value'], 'entity-type'):
                                if elm['datavalue']['value']['entity-type'] == 'item':
                                    if 'id' in elm['datavalue']['value'].keys():
                                        q = elm['datavalue']['value']['id']
                                        property_tuples.append((p, q))

    return property_tuples

def has_key(d, key):
    """
    check if d is a dict and has key k. Otherwise return False
    :param d:
    :return:
    """
    if isinstance(d, dict):
        if key in d.keys():
            return True
    return False

def get_parent_category(j):
    """
    if the concept has a subcategory property, return the parent
    subcategory is P279
    :param j:
    :return:
    """
    parents = []
    if 'claims' in j.keys():
        for p, r in j['claims'].items():
            for elm in r:
                if 'mainsnak' in elm.keys():
                    if 'property' in elm['mainsnak'].keys():
                        p = elm['mainsnak']['property']
                        if p == 'P279':
                            if 'datavalue' in elm['mainsnak'].keys():
                                if 'value' in elm['mainsnak']['datavalue'].keys():
                                    if 'id' in elm['mainsnak']['datavalue']['value'].keys():
                                        q = elm['mainsnak']['datavalue']['value']['id']
                                        parents.append('{}{}'.format(SUBCAT, q))
    if len(parents) > 0:
        return parents
    return None

def get_wikipage_title(j, lang):
    """
    Returns title of wikipage, or None if concept does not have a wikipage in lang
    :param j:
    :param lang:
    :return:
    """
    if has_key(j, 'sitelinks'):
        wikikey = '{}wiki'.format(lang)
        if has_key(j['sitelinks'], wikikey):
            if has_key(j['sitelinks'][wikikey], 'title'):
                return j['sitelinks'][wikikey]['title']
    return None


def get_aliases_descriptions(qid, lang, dump, pointers):
    """
    retrieve aliases and descriptions of q from wikidata dump
    :param q:
    :return:
    """
    try:
        j = get_q_json(dump, pointers[qid])
        aliases = get_aliases(j, lang)
        description = get_description(j, lang)
        return {'aliases': aliases, 'description': description}
    except KeyError:
        print('{} is not present in dump'.format(qid))
        return None

def get_longest_alias(title, aliases):
    """
    get longest sequence, where length is defined as number of tokens separated by whitespace
    """
    if aliases is None or aliases['aliases'] is None: return title
    aliases = aliases['aliases']
    # get longest alias
    longest = aliases[0]
    if len(aliases) > 1:
        for elm in aliases[1:]:
            if len(elm.split()) > len(longest.split()):
                longest = elm
    if len(longest.split()) > len(title.split()):
        return longest
    else: return title


def augment_data(analogy_file, outfile, lang, pointers_file, dump_file, setting='longest'):
    """
    if setting == longest, only retrieve longest alias for each entity
    """
    pointers = load_pointers(pointers_file)
    dump = file_open(dump_file)
    with open(outfile, 'w') as f:

        writer = csv.DictWriter(f, delimiter=';', fieldnames = ['Q1', 'Q1_id',  'Q2', 'Q2_id', 'Q3', 'Q3_id','Q4', 'Q4_id', 'distance', 'distance_pairwise'])
        for row in read_analogy_data(analogy_file):
            if not is_comment(row):
                if setting == 'longest':
                    q1_context = get_longest_alias(row['Q1'], get_aliases_descriptions(row['Q1_id'], lang, dump, pointers))
                    q2_context = get_longest_alias(row['Q2'], get_aliases_descriptions(row['Q2_id'], lang, dump, pointers))
                    q3_context = get_longest_alias(row['Q3'], get_aliases_descriptions(row['Q3_id'], lang, dump, pointers))
                    q4_context = get_longest_alias(row['Q4'], get_aliases_descriptions(row['Q4_id'], lang, dump, pointers))
                else:
                    q1_context = get_aliases_descriptions(row['Q1_id'], lang, dump, pointers)
                    q2_context = get_aliases_descriptions(row['Q2_id'], lang, dump, pointers)
                    q3_context = get_aliases_descriptions(row['Q3_id'], lang, dump, pointers)
                    q4_context = get_aliases_descriptions(row['Q4_id'], lang, dump, pointers)


                row['Q1'] = q1_context
                row['Q2'] = q2_context
                row['Q3'] = q3_context
                row['Q4'] = q4_context

                print(row)
                writer.writerow(row)
            else:
                writer.writerow(row)
    f.close()

if __name__=="__main__":
    cfg = 'config.cfg'
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(cfg)

    dump_path = config.get('Files', 'wikidata_dump')
    pointers_path = config.get('Files', 'wikidata_pointers')

    langs = ['da', 'de', 'en', 'es', 'fi', 'fr', 'it', 'nl', 'pl', 'pt', 'sv']
    for setting in ['unique', 'all']:
        for lang in langs:
            fname = os.path.join(config.get('Files', 'data'), 'analogy_{}_{}_dists.csv'.format(setting, lang))
            fname_out = os.path.join(config.get('Files', 'data'), 'analogy_{}_{}_longestalias.csv'.format(setting, lang))
            augment_data(analogy_file=fname, outfile=fname_out, lang=lang, pointers_file=pointers_path, dump_file=dump_path, setting='longest')