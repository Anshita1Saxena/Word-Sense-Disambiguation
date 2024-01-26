import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords as sw

import string
from nltk.stem import WordNetLemmatizer


# lemmatizer instance Initialization
lematizer = WordNetLemmatizer()

stopWords = set(sw.words('english'))

tagDict = {
    'N': wn.NOUN,
    'V': wn.VERB,
    'R': wn.ADV,
    'J': wn.ADJ,
    'S': wn.ADJ_SAT
}
ss_type = {
    '1': wn.NOUN,
    '2': wn.VERB,
    '3': wn.ADV,
    '4': wn.ADJ,
    '5': wn.ADJ_SAT
}


def attach_pos_tag(tag):
    """
    attach the tag to the lemma.
    """
    return tagDict[tag[0]] if (tag[0] in tagDict) else "OTHER_TAG"


def preprocess(text):
    """
    Preprocess the text based on lemmatization, stopwords, and punctuation.
    """
    final_text = []
    for sentence in text:
        tokenized_text = nltk.word_tokenize(sentence)
        final_text.append(' '.join([lematizer.lemmatize(word) for word in tokenized_text
                                    if word not in stopWords
                                    and word not in string.punctuation]))
    return final_text


def to_synset(s):
    # Take the true sense from the provided key.
    return wn.lemma_from_key(s).synset()


def check_accuracy(true_values, pred_values):
    """
    Check the accuracy between the true values and the predicted values.
    """
    correct_values = 0
    total = len(true_values)
    if total != len(pred_values):
        print(f"Mismatch in true and predicted counts. Found {total} in true and {len(pred_values)} in predicted.")
    for (true_value, pred_value) in zip(true_values, pred_values):
        if pred_value is None:
            continue
        if pred_value in true_value:
            correct_values += 1
    return correct_values/total


def check_true_word_senses(key):
    """
    Retrieves all the true word sense from the key.
    """
    true_word_senses = []
    for (k, true_values) in key.items():
        true_word_sense = []
        for true_value in true_values:
            true_word_sense.append(to_synset(true_value))
        true_word_senses.append(true_word_sense)
    return true_word_senses
