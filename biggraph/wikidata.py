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

def augment_data(analogy_file, outfile, lang, pointers_file, dump_file):
    pointers = load_pointers(pointers_file)
    dump = file_open(dump_file)
    with open(outfile, 'w') as f:
        writer = csv.writer(f, delimiter=';')
        for row in read_analogy_data(analogy_file):
            if not is_comment(row):
                q1_context = get_aliases_descriptions(row['Q1_id'], lang, dump, pointers)
                q2_context = get_aliases_descriptions(row['Q2_id'], lang, dump, pointers)
                q3_context = get_aliases_descriptions(row['Q3_id'], lang, dump, pointers)
                q4_context = get_aliases_descriptions(row['Q4_id'], lang, dump, pointers)
                row.update({'Q1_context': q1_context})
                row.update({'Q2_context': q2_context})
                row.update({'Q3_context': q3_context})
                row.update({'Q4_context': q4_context})
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
    for lang in langs:
        fname = os.path.join(config.get('Files', 'data'), 'analogy_all_{}.csv'.format(lang))
        fname_out = os.path.join(config.get('Files', 'data'), 'analogy_all_{}_contexts.csv'.format(lang))
        augment_data(analogy_file=fname, outfile=fname_out, lang=lang, pointers_file=pointers_path, dump_file=dump_path)