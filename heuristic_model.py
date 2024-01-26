from utils import *
from nltk.wsd import lesk
from sentence_transformers import SentenceTransformer, util


def check_most_frequent_word_senses(key, instances):
    """
    # Retrieves all the most frequent word sense from the instance.
    """
    most_frequent_word_senses = []
    for (k, v) in key.items():
        instance_lemma = instances[k].lemma
        instance_pos = instances[k].pos
        # Wordnet orders the synset by its estimated frequency of usage
        # https://wordnet.princeton.edu/documentation/wn1wn
        most_frequent_word_senses.append(wn.synsets(instance_lemma, tagDict[instance_pos.upper()])[0])
    return most_frequent_word_senses


def most_frequent_word_sense_baseline(key, instances):
    """
    Returns the accuracy for the most frequent word sense.
    """
    # Check True Word Sense
    true_word_senses = check_true_word_senses(key)
    # Check Most Frequent Word Sense
    pred_word_senses = check_most_frequent_word_senses(key, instances)
    # Calculate Accuracy
    most_frequent_sense_accuracy = check_accuracy(true_word_senses, pred_word_senses)
    return most_frequent_sense_accuracy


def check_lesk_word_senses(key, instances):
    """
    Retrieves all the word sense from the instance by applying NLTK Lesk Algorithm.
    """
    lesk_word_senses = []
    for (k, v) in key.items():
        instance_context_sentence = instances[k].context
        instance_lemma = instances[k].lemma
        instance_pos = instances[k].pos
        word_sense = lesk(instance_context_sentence, instance_lemma, instance_pos,
                          wn.synsets(v[0].split("%")[0], ss_type[v[0].split("%")[-1][0]]))
        lesk_word_senses.append(word_sense)
    return lesk_word_senses


def lesk_word_sense_baseline(key, instances):
    """
    Returns the accuracy for the nltk lesk algorithm.
    """
    # Check True Word Sense
    true_word_senses = check_true_word_senses(key)
    # Check Lesk Word Sense
    pred_word_senses = check_lesk_word_senses(key, instances)
    # Calculate Accuracy
    lesk_word_sense_accuracy = check_accuracy(true_word_senses, pred_word_senses)
    return lesk_word_sense_accuracy


def custom(context_sentence, ambiguous_word, pos=None, lemmatize=False, stopwords=False, punctuation=False, cosine=False):
    """
    Custom model based on Lesk Intuition
    Context sentences, ambiguous word, part of speech, lemmatization, stopwords, and punctuation on word definition,
    and cosine similarity as arguments for choice on user to select model
    """
    max_overlaps = 0
    custom_sense = None
    # custom model for cosine similarity check from huggingface
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # retrieve all the senses of the ambiguous word and allocate the sense based on maximum overlap
    for ss in wn.synsets(ambiguous_word):
        if pos and ss.pos() != pos:
            continue

        custom_dictionary = []

        # Includes definition
        custom_dictionary += ss.definition().split()
        # Includes lemma_names
        custom_dictionary += ss.lemma_names()
        context_sentence = [i for i in context_sentence]
        # print(context_sentence)

        # If the source context is lematized then the dictionary sentences should also lematized
        if lemmatize == True:
            custom_dictionary = [lematizer.lemmatize(i) for i in custom_dictionary]

        # If the source context is stopwords free then the dictionary sentences should also stopwords
        if stopwords == True:
            custom_dictionary = [i for i in custom_dictionary if i not in stopWords]

        # If the source context is punctuation free then the dictionary sentences should also punctuation free
        if punctuation == True:
            custom_dictionary = [i for i in custom_dictionary if i not in string.punctuation]

        if cosine:
            custom_dictionary = ' '.join(custom_dictionary)
            context_sentence = ''.join(context_sentence)

            # create the embedding
            embedding_1 = model.encode(custom_dictionary, convert_to_tensor=True)
            embedding_2 = model.encode(context_sentence, convert_to_tensor=True)

            # find the angle between the embeddings
            overlaps = util.pytorch_cos_sim(embedding_1, embedding_2)
            # Assign sense based on maximum angle chosen
            if overlaps > max_overlaps:
                custom_sense = ss
                max_overlaps = overlaps
        else:
            overlaps = set(custom_dictionary).intersection(context_sentence)
            # Assign sense based on maximum counts of the same words
            if len(overlaps) > max_overlaps:
                custom_sense = ss
                max_overlaps = len(overlaps)

    return custom_sense


def check_custom_word_senses(key, instances, lemmatize, stopwords, punctuation, cosine):
    """
    Retrieves all the senses based on the overlap from the instance
    """
    custom_word_senses = []
    for (k, v) in key.items():
        instance_lemma = instances[k].lemma
        instance_pos = instances[k].pos
        instance_context = instances[k].context
        custom_word_senses.append(custom(instance_context, instance_lemma, instance_pos, lemmatize, stopwords, punctuation, cosine))
    return custom_word_senses


def custom_word_sense_model(key, instances, lemmatize, stopwords, punctuation, cosine):
    """
    Return the accuracy of the custom sense model
    """
    # Check True Word Sense
    true_word_senses = check_true_word_senses(key)
    # Check Most Frequent Word Sense
    pred_word_senses = check_custom_word_senses(key, instances, lemmatize, stopwords, punctuation, cosine)
    # Calculate Accuracy
    custom_sense_accuracy = check_accuracy(true_word_senses, pred_word_senses)
    return custom_sense_accuracy
