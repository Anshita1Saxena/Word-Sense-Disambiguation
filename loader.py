"""
@author: jcheung

Developed for Python 2. Automatically converted to Python 3; may result in bugs.
"""
import xml.etree.cElementTree as ET
import codecs

from heuristic_model import *
from bootstrapping_yarowsky_model import *

# lemmatizer instance Initialization
lematizer = WordNetLemmatizer()

stopWords = set(sw.words('english'))


class WSDInstance:
    def __init__(self, my_id, lemma, context, index, pos):
        self.id = my_id  # id of the WSD instance
        self.lemma = lemma.decode("utf-8")  # lemma of the word whose sense is to be resolved
        self.context = context  # lemma of all the words in the sentential context
        self.index = index  # index of lemma within the context
        self.pos = pos  # part of speech

    def __str__(self):
        """
        For printing purposes.
        """
        return self.id + "\t" + self.lemma + "\t" + ''.join(str(self.context)) \
               + "\t" + str(self.index)


def load_instances(f, lemmatize=False, stopwords=False, punctuation=False):
    """
    Load two lists of cases to perform WSD on. The structure that is returned is a dict, where
    the keys are the ids, and the values are instances of WSDInstance.
    """
    tree = ET.parse(f)
    root = tree.getroot()

    dev_instances = {}
    test_instances = {}

    for text in root:
        if text.attrib['id'].startswith('d001'):
            instances = dev_instances
        else:
            instances = test_instances
        for sentence in text:
            if not lemmatize and not stopwords and not punctuation:
                context = [to_ascii(el.attrib['lemma']).decode("utf-8") for el in sentence]
            if lemmatize and not stopwords and not punctuation:
                context = [lematizer.lemmatize(to_ascii(el.attrib['lemma']).decode("utf-8")) for el in sentence]
            if lemmatize and stopwords and not punctuation:
                context = [lematizer.lemmatize(to_ascii(el.attrib['lemma']).decode("utf-8")) for el in sentence
                           if to_ascii(el.attrib['lemma']).decode("utf-8") not in stopWords]
            if lemmatize and not stopwords and punctuation:
                context = [lematizer.lemmatize(to_ascii(el.attrib['lemma']).decode("utf-8")) for el in sentence
                           if to_ascii(el.attrib['lemma']).decode("utf-8") not in string.punctuation]
            if not lemmatize and stopwords and not punctuation:
                context = [to_ascii(el.attrib['lemma']).decode("utf-8") not in stopWords for el in sentence]
            if not lemmatize and not stopwords and punctuation:
                context = [to_ascii(el.attrib['lemma']).decode("utf-8") not in string.punctuation for el in sentence]
            if not lemmatize and stopwords and punctuation:
                context = [to_ascii(el.attrib['lemma']).decode("utf-8") not in stopWords for el in sentence
                           if to_ascii(el.attrib['lemma']).decode("utf-8") not in string.punctuation]
            if lemmatize and stopwords and punctuation:
                context = [lematizer.lemmatize(to_ascii(el.attrib['lemma']).decode("utf-8")) for el in sentence
                           if to_ascii(el.attrib['lemma']).decode("utf-8") not in stopWords
                           and to_ascii(el.attrib['lemma']).decode("utf-8") not in string.punctuation]
            for i, el in enumerate(sentence):
                if el.tag == 'instance':
                    my_id = el.attrib['id']
                    lemma = to_ascii(el.attrib['lemma'])
                    pos = attach_pos_tag(el.attrib['pos'])
                    instances[my_id] = WSDInstance(my_id, lemma, context, i, pos)
    return dev_instances, test_instances


def load_key(f):
    """
    Load the solutions as dicts.
    Key is the id
    Value is the list of correct sense keys.
    """
    dev_key = {}
    test_key = {}
    for line in open(f):
        if len(line) <= 1: continue
        doc, my_id, sense_key = line.strip().split(' ', 2)
        if doc == 'd001':
            dev_key[my_id] = sense_key.split()
        else:
            test_key[my_id] = sense_key.split()
    return dev_key, test_key


def to_ascii(s):
    # remove all non-ascii characters.
    return codecs.encode(s, 'ascii', 'ignore')


if __name__ == '__main__':
    data_f = 'multilingual-all-words.en.xml'
    key_f = 'wordnet.en.key'
    dev_key, test_key = load_key(key_f)

    # Arguments for recording experiments
    # simple load instances
    dev_instances, test_instances = load_instances(data_f)
    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
    dev_instances = {k: v for (k, v) in dev_instances.items() if k in dev_key}
    test_instances = {k: v for (k, v) in test_instances.items() if k in test_key}
    # simple load instances with lemmatize
    lem_dev_instances, lem_test_instances = load_instances(data_f, True, False, False)
    lem_dev_instances = {k: v for (k, v) in lem_dev_instances.items() if k in dev_key}
    lem_test_instances = {k: v for (k, v) in lem_test_instances.items() if k in test_key}
    # simple load instances with lemmatize and stopwords
    ls_dev_instances, ls_test_instances = load_instances(data_f, True, True, False)
    ls_dev_instances = {k: v for (k, v) in ls_dev_instances.items() if k in dev_key}
    ls_test_instances = {k: v for (k, v) in ls_test_instances.items() if k in test_key}
    # simple load instances with lemmatize and punctuation
    lp_dev_instances, lp_test_instances = load_instances(data_f, True, False, True)
    lp_dev_instances = {k: v for (k, v) in lp_dev_instances.items() if k in dev_key}
    lp_test_instances = {k: v for (k, v) in lp_test_instances.items() if k in test_key}
    # simple load instances with stopwords
    s_dev_instances, s_test_instances = load_instances(data_f, False, True, False)
    s_dev_instances = {k: v for (k, v) in s_dev_instances.items() if k in dev_key}
    s_test_instances = {k: v for (k, v) in s_test_instances.items() if k in test_key}
    # simple load instances with punctuation
    p_dev_instances, p_test_instances = load_instances(data_f, False, False, True)
    p_dev_instances = {k: v for (k, v) in p_dev_instances.items() if k in dev_key}
    p_test_instances = {k: v for (k, v) in p_test_instances.items() if k in test_key}
    # simple load instances with stopwords and punctuation
    sp_dev_instances, sp_test_instances = load_instances(data_f, False, True, True)
    sp_dev_instances = {k: v for (k, v) in sp_dev_instances.items() if k in dev_key}
    sp_test_instances = {k: v for (k, v) in sp_test_instances.items() if k in test_key}
    # simple load instances with lemmatize, stopwords and punctuation
    lsp_dev_instances, lsp_test_instances = load_instances(data_f, True, True, True)
    lsp_dev_instances = {k: v for (k, v) in lsp_dev_instances.items() if k in dev_key}
    lsp_test_instances = {k: v for (k, v) in lsp_test_instances.items() if k in test_key}

    results_file = open('experiment_results.txt', 'w')
    results_file.write(
        f"Dev Most Frequent Accuracy without any settings: {most_frequent_word_sense_baseline(dev_key, dev_instances)}")
    results_file.write("\n")
    results_file.write(
        f"Dev Most Frequent Accuracy with lemmatization: {most_frequent_word_sense_baseline(dev_key, lem_dev_instances)}")
    results_file.write("\n")
    results_file.write(
        f"Dev Most Frequent Accuracy with lemmatization and stopwords: {most_frequent_word_sense_baseline(dev_key, ls_dev_instances)}")
    results_file.write("\n")
    results_file.write(
        f"Dev Most Frequent Accuracy with lemmatize and punctuation: {most_frequent_word_sense_baseline(dev_key, lp_dev_instances)}")
    results_file.write("\n")
    results_file.write(
        f"Dev Most Frequent Accuracy with stopwords: {most_frequent_word_sense_baseline(dev_key, s_dev_instances)}")
    results_file.write("\n")
    results_file.write(
        f"Dev Most Frequent Accuracy with punctuation: {most_frequent_word_sense_baseline(dev_key, p_dev_instances)}")
    results_file.write("\n")
    results_file.write(
        f"Dev Most Frequent Accuracy with stopwords and punctuation: {most_frequent_word_sense_baseline(dev_key, sp_dev_instances)}")
    results_file.write("\n")
    results_file.write(
        f"Dev Most Frequent Accuracy with lemmatize, stopwords and punctuation: {most_frequent_word_sense_baseline(dev_key, lsp_dev_instances)}")
    results_file.write("\n")
    results_file.write(
        f"Test Most Frequent Accuracy without any settings: {most_frequent_word_sense_baseline(test_key, test_instances)}")
    results_file.write("\n")
    results_file.write(
        f"Test Most Frequent Accuracy with lemmatization: {most_frequent_word_sense_baseline(test_key, lem_test_instances)}")
    results_file.write("\n")
    results_file.write(
        f"Test Most Frequent Accuracy with lemmatization and stopwords: {most_frequent_word_sense_baseline(test_key, ls_test_instances)}")
    results_file.write("\n")
    results_file.write(
        f"Test Most Frequent Accuracy with lemmatize and punctuation: {most_frequent_word_sense_baseline(test_key, lp_test_instances)}")
    results_file.write("\n")
    results_file.write(
        f"Test Most Frequent Accuracy with stopwords: {most_frequent_word_sense_baseline(test_key, s_test_instances)}")
    results_file.write("\n")
    results_file.write(
        f"Test Most Frequent Accuracy with punctuation: {most_frequent_word_sense_baseline(test_key, p_test_instances)}")
    results_file.write("\n")
    results_file.write(
        f"Test Most Frequent Accuracy with stopwords and punctuation: {most_frequent_word_sense_baseline(test_key, sp_test_instances)}")
    results_file.write("\n")
    results_file.write(
        f"Test Most Frequent Accuracy with lemmatize, stopwords and punctuation: {most_frequent_word_sense_baseline(test_key, lsp_test_instances)}")
    results_file.write("\n")
    results_file.write(
        f"Dev Lesk Frequent Accuracy without any settings: {lesk_word_sense_baseline(dev_key, dev_instances)}")
    results_file.write("\n")
    results_file.write(
        f"Dev Lesk Frequent Accuracy with lemmatization: {lesk_word_sense_baseline(dev_key, lem_dev_instances)}")
    results_file.write("\n")
    results_file.write(
        f"Dev Lesk Frequent Accuracy with lemmatization and stopwords: {lesk_word_sense_baseline(dev_key, ls_dev_instances)}")
    results_file.write("\n")
    results_file.write(
        f"Dev Lesk Frequent Accuracy with lemmatize and punctuation: {lesk_word_sense_baseline(dev_key, lp_dev_instances)}")
    results_file.write("\n")
    results_file.write(
        f"Dev Lesk Frequent Accuracy with stopwords: {lesk_word_sense_baseline(dev_key, s_dev_instances)}")
    results_file.write("\n")
    results_file.write(
        f"Dev Lesk Frequent Accuracy with punctuation: {lesk_word_sense_baseline(dev_key, p_dev_instances)}")
    results_file.write("\n")
    results_file.write(
        f"Dev Lesk Frequent Accuracy with stopwords and punctuation: {lesk_word_sense_baseline(dev_key, sp_dev_instances)}")
    results_file.write("\n")
    results_file.write(
        f"Dev Lesk Frequent Accuracy with lemmatize, stopwords and punctuation: {lesk_word_sense_baseline(dev_key, lsp_dev_instances)}")
    results_file.write("\n")
    results_file.write(
        f"Dev Lesk Frequent Accuracy with lemmatization: {lesk_word_sense_baseline(test_key, lem_test_instances)}")
    results_file.write("\n")
    results_file.write(
        f"Test Lesk Frequent Accuracy without any settings: {lesk_word_sense_baseline(test_key, test_instances)}")
    results_file.write("\n")
    results_file.write(
        f"Test Lesk Frequent Accuracy with stopwords: {lesk_word_sense_baseline(test_key, s_test_instances)}")
    results_file.write("\n")
    results_file.write(
        f"Test Lesk Frequent Accuracy with punctuation: {lesk_word_sense_baseline(test_key, p_test_instances)}")
    results_file.write("\n")
    results_file.write(
        f"Test Lesk Frequent Accuracy with stopwords and punctuation: {lesk_word_sense_baseline(test_key, sp_test_instances)}")
    results_file.write("\n")
    results_file.write(
        f"Dev Custom Word Accuracy without any settings with word intersection: {custom_word_sense_model(dev_key, dev_instances, False, False, False, False)}")
    results_file.write("\n")
    results_file.write(
        f"Dev Custom Word Accuracy without any settings with word intersection: {custom_word_sense_model(dev_key, dev_instances, False, False, False, False)}")
    results_file.write("\n")
    results_file.write(
        f"Dev Custom Word Accuracy without any settings with cosine similarity: {custom_word_sense_model(dev_key, dev_instances, False, False, False, True)}")
    results_file.write("\n")
    results_file.write(
        f"Dev Custom Word Accuracy without any settings with cosine similarity: {custom_word_sense_model(dev_key, dev_instances, False, False, False, True)}")
    results_file.write("\n")
    results_file.write(
        f"Dev Custom Word Accuracy with all settings with word intersection: {custom_word_sense_model(dev_key, lsp_dev_instances, True, True, True, False)}")
    results_file.write("\n")
    results_file.write(
        f"Dev Custom Word Accuracy with all settings with word intersection: {custom_word_sense_model(dev_key, lsp_dev_instances, True, True, True, False)}")
    results_file.write("\n")
    results_file.write(
        f"Dev Custom Word Accuracy with all settings with cosine similarity: {custom_word_sense_model(dev_key, lsp_dev_instances, True, True, True, True)}")
    results_file.write("\n")
    results_file.write(
        f"Dev Custom Word Accuracy with all settings with cosine similarity: {custom_word_sense_model(dev_key, lsp_dev_instances, True, True, True, True)}")
    results_file.write("\n")
    results_file.write(
        f"Test Custom Word Accuracy with all settings with word intersection: {custom_word_sense_model(test_key, lsp_test_instances, True, True, True, False)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'sense' Accuracy with all settings, with 3 iterations and confidence = 0.9[dev accuracy, test accuracy]: {bootstrapping(dev_key, lsp_dev_instances, test_key, lsp_test_instances, 3, 'sense', 0.9, True, False)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'sense' Accuracy with all settings, with 4 iterations and confidence = 0.9[dev accuracy, test accuracy]: {bootstrapping(dev_key, lsp_dev_instances, test_key, lsp_test_instances, 4, 'sense', 0.9, True, False)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'sense' Accuracy with all settings, with 5 iterations and confidence = 0.9[dev accuracy, test accuracy]: {bootstrapping(dev_key, lsp_dev_instances, test_key, lsp_test_instances, 5, 'sense', 0.9, True, False)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'sense' Accuracy with all settings, with 4 iterations and confidence = 0.9[dev accuracy, test accuracy]: {bootstrapping(dev_key, lsp_dev_instances, test_key, lsp_test_instances, 4, 'sense', 0.9, True, True)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'sense' Accuracy without preprocessing, with 3 iterations and confidence = 0.9[dev accuracy, test accuracy]: {bootstrapping(dev_key, dev_instances, test_key, test_instances, 3, 'sense', 0.9, False, False)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'sense' Accuracy without preprocessing, with 4 iterations and confidence = 0.9[dev accuracy, test accuracy]: {bootstrapping(dev_key, dev_instances, test_key, test_instances, 4, 'sense', 0.9, False, False)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'sense' Accuracy without preprocessing, with 5 iterations and confidence = 0.9[dev accuracy, test accuracy]: {bootstrapping(dev_key, dev_instances, test_key, test_instances, 5, 'sense', 0.9, False, False)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'sense' Accuracy without preprocessing, with 4 iterations and confidence = 0.9[dev accuracy, test accuracy]: {bootstrapping(dev_key, dev_instances, test_key, test_instances, 4, 'sense', 0.9, False, True)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'time' Accuracy with all settings, with 7 iterations and confidence = 0.9[dev accuracy, test accuracy]: {bootstrapping(dev_key, lsp_dev_instances, test_key, lsp_test_instances, 7, 'time', 0.9, True, False)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'time' Accuracy with all settings, with 8 iterations and confidence = 0.9[dev accuracy, test accuracy]: {bootstrapping(dev_key, lsp_dev_instances, test_key, lsp_test_instances, 8, 'time', 0.9, True, False)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'time' Accuracy with all settings, with 9 iterations and confidence = 0.9[dev accuracy, test accuracy]: {bootstrapping(dev_key, lsp_dev_instances, test_key, lsp_test_instances, 9, 'time', 0.9, True, False)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'time' Accuracy with all settings, with 8 iterations and confidence = 0.9[dev accuracy, test accuracy]: {bootstrapping(dev_key, lsp_dev_instances, test_key, lsp_test_instances, 8, 'time', 0.9, True, True)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'time' Accuracy without preprocessing, with 4 iterations and confidence = 0.9[dev accuracy, test accuracy]: {bootstrapping(dev_key, dev_instances, test_key, test_instances, 4, 'time', 0.9, False, False)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'time' Accuracy without preprocessing, with 5 iterations and confidence = 0.9[dev accuracy, test accuracy]: {bootstrapping(dev_key, dev_instances, test_key, test_instances, 5, 'time', 0.9, False, False)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'time' Accuracy without preprocessing, with 6 iterations and confidence = 0.9[dev accuracy, test accuracy]: {bootstrapping(dev_key, dev_instances, test_key, test_instances, 6, 'time', 0.9, False, False)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'time' Accuracy without preprocessing, with 5 iterations and confidence = 0.9[dev accuracy, test accuracy]: {bootstrapping(dev_key, dev_instances, test_key, test_instances, 5, 'time', 0.9, False, True)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'level' Accuracy with all settings, with 2 iterations and confidence = 0.7[dev accuracy, test accuracy]: {bootstrapping(dev_key, lsp_dev_instances, test_key, lsp_test_instances, 2, 'level', 0.7, True, False)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'level' Accuracy with all settings, with 3 iterations and confidence = 0.7[dev accuracy, test accuracy]: {bootstrapping(dev_key, lsp_dev_instances, test_key, lsp_test_instances, 3, 'level', 0.7, True, False)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'level' Accuracy with all settings, with 4 iterations and confidence = 0.7[dev accuracy, test accuracy]: {bootstrapping(dev_key, lsp_dev_instances, test_key, lsp_test_instances, 4, 'level', 0.7, True, False)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'level' Accuracy with all settings, with 3 iterations and confidence = 0.7[dev accuracy, test accuracy]: {bootstrapping(dev_key, lsp_dev_instances, test_key, lsp_test_instances, 3, 'level', 0.7, True, True)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'level' Accuracy without preprocessing, with 2 iterations and confidence = 0.7[dev accuracy, test accuracy]: {bootstrapping(dev_key, dev_instances, test_key, test_instances, 2, 'level', 0.9, False, False)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'level' Accuracy without preprocessing, with 3 iterations and confidence = 0.7[dev accuracy, test accuracy]: {bootstrapping(dev_key, dev_instances, test_key, test_instances, 3, 'level', 0.9, False, False)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'level' Accuracy without preprocessing, with 4 iterations and confidence = 0.7[dev accuracy, test accuracy]: {bootstrapping(dev_key, dev_instances, test_key, test_instances, 4, 'level', 0.9, False, False)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'level' Accuracy without preprocessing, with 3 iterations and confidence = 0.7[dev accuracy, test accuracy]: {bootstrapping(dev_key, dev_instances, test_key, test_instances, 3, 'level', 0.9, False, True)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'life' Accuracy with all settings, with 1 iterations and confidence = 0.9[dev accuracy, test accuracy]: {bootstrapping(dev_key, lsp_dev_instances, test_key, lsp_test_instances, 1, 'life', 0.9, True, False)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'life' Accuracy with all settings, with 2 iterations and confidence = 0.9[dev accuracy, test accuracy]: {bootstrapping(dev_key, lsp_dev_instances, test_key, lsp_test_instances, 2, 'life', 0.9, True, False)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'life' Accuracy with all settings, with 3 iterations and confidence = 0.9[dev accuracy, test accuracy]: {bootstrapping(dev_key, lsp_dev_instances, test_key, lsp_test_instances, 3, 'life', 0.9, True, False)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'life' Accuracy with all settings, with 2 iterations and confidence = 0.9[dev accuracy, test accuracy]: {bootstrapping(dev_key, lsp_dev_instances, test_key, lsp_test_instances, 2, 'life', 0.9, True, True)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'life' Accuracy without preprocessing, with 3 iterations and confidence = 0.9[dev accuracy, test accuracy]: {bootstrapping(dev_key, dev_instances, test_key, test_instances, 3, 'life', 0.8, False, False)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'life' Accuracy without preprocessing, with 4 iterations and confidence = 0.9[dev accuracy, test accuracy]: {bootstrapping(dev_key, dev_instances, test_key, test_instances, 4, 'life', 0.8, False, False)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'life' Accuracy without preprocessing, with 5 iterations and confidence = 0.9[dev accuracy, test accuracy]: {bootstrapping(dev_key, dev_instances, test_key, test_instances, 5, 'life', 0.8, False, False)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'life' Accuracy without preprocessing, with 4 iterations and confidence = 0.9[dev accuracy, test accuracy]: {bootstrapping(dev_key, dev_instances, test_key, test_instances, 4, 'life', 0.8, False, True)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'deal' Accuracy with all settings, with 1 iterations and confidence = 0.9[dev accuracy, test accuracy]: {bootstrapping(dev_key, lsp_dev_instances, test_key, lsp_test_instances, 1, 'deal', 0.9, True, False)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'deal' Accuracy with all settings, with 2 iterations and confidence = 0.9[dev accuracy, test accuracy]: {bootstrapping(dev_key, lsp_dev_instances, test_key, lsp_test_instances, 2, 'deal', 0.9, True, False)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'deal' Accuracy with all settings, with 3 iterations and confidence = 0.9[dev accuracy, test accuracy]: {bootstrapping(dev_key, lsp_dev_instances, test_key, lsp_test_instances, 3, 'deal', 0.9, True, False)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'deal' Accuracy with all settings, with 2 iterations and confidence = 0.9[dev accuracy, test accuracy]: {bootstrapping(dev_key, lsp_dev_instances, test_key, lsp_test_instances, 2, 'deal', 0.9, True, True)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'deal' Accuracy without preprocessing, with 2 iterations and confidence = 0.9[dev accuracy, test accuracy]: {bootstrapping(dev_key, dev_instances, test_key, test_instances, 2, 'deal', 0.9, False, False)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'deal' Accuracy without preprocessing, with 3 iterations and confidence = 0.9[dev accuracy, test accuracy]: {bootstrapping(dev_key, dev_instances, test_key, test_instances, 3, 'deal', 0.9, False, False)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'deal' Accuracy without preprocessing, with 4 iterations and confidence = 0.9[dev accuracy, test accuracy]: {bootstrapping(dev_key, dev_instances, test_key, test_instances, 4, 'deal', 0.9, False, False)}")
    results_file.write("\n")
    results_file.write(
        f"Bootstrapping Word for word 'deal' Accuracy without preprocessing, with 3 iterations and confidence = 0.9[dev accuracy, test accuracy]: {bootstrapping(dev_key, dev_instances, test_key, test_instances, 3, 'deal', 0.9, False, True)}")
    results_file.close()
