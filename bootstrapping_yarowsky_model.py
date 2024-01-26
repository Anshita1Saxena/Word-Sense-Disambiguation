import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB as MNB
from utils import *


def load_word_data(filename, sense1, sense2, preprocess_data=False):
    """
    Load data for words time, power, level, and sense.
    """
    sense_df = pd.read_csv(filename, names=['context', 'sense'])
    sense_df.drop_duplicates(inplace=True)
    # if filename == "final_deal.csv":
    #     sense_df = sense_df.sample(frac=1).reset_index(drop=True)
    sense_1_df = sense_df[sense_df['sense'] == sense1].iloc[:10]
    sense_2_df = sense_df[sense_df['sense'] == sense2].iloc[:10]
    final_small_annotated_data = pd.concat([sense_1_df, sense_2_df], axis=0, ignore_index=True)
    sense_df = pd.concat([sense_df, final_small_annotated_data]).drop_duplicates(keep=False)
    sense_df.reset_index(inplace=True, drop='index')
    # print(sense_df)
    if preprocess_data:
        final_small_annotated_data = pd.concat([pd.DataFrame(preprocess(final_small_annotated_data['context']),
                                                             columns=['context']),
                                                final_small_annotated_data['sense']], axis=1)
        data_df = pd.concat([pd.DataFrame(preprocess(sense_df['context']), columns=['context']), sense_df['sense']], axis=1)
    else:
        final_small_annotated_data = pd.concat([pd.DataFrame(final_small_annotated_data['context'],
                                                             columns=['context']),
                                                final_small_annotated_data['sense']], axis=1)
        data_df = pd.concat([pd.DataFrame(sense_df['context'], columns=['context']), sense_df['sense']], axis=1)
    return final_small_annotated_data, data_df


def check_model_accuracy(final_small_annotated_data, data_df, dev_filtered_df, test_filtered_df, iterations_number, confidence, preprocess_data=False, test=False):
    """
    Return the final accuracy of bootstrapping model
    """
    accuracy = []
    dev_accuracy = 0
    if preprocess_data:
        dev_filtered_df = dev_filtered_df.copy(deep=True)
        test_filtered_df = test_filtered_df.copy(deep=True)
        dev_filtered_df['context'] = preprocess(dev_filtered_df['context'])
        test_filtered_df['context'] = preprocess(test_filtered_df['context'])

    iterations = iterations_number
    for i in range(iterations):
        # Step 3: Train a supervised learning algorithm from the seed set
        vectorizer = CountVectorizer()
        X_train = vectorizer.fit_transform(final_small_annotated_data['context'])
        y_train = final_small_annotated_data['sense']

        classifier = MNB()
        classifier.fit(X_train, y_train)

        # Step 3 (continued): Apply the supervised model to the entire data set
        X_all = vectorizer.transform(data_df['context'])
        predictions = classifier.predict(X_all)
        prediction_df = pd.DataFrame(predictions, columns=['sense'])

        # Keep the highly confident classification outputs to be the new seed set
        confidence_threshold = confidence
        high_confidence_indices = np.where(np.max(classifier.predict_proba(X_all), axis=1) > confidence_threshold)[0]

        new_seed_data = data_df.iloc[high_confidence_indices]
        new_seed_sense = prediction_df.iloc[high_confidence_indices]
        new_seed_df = pd.concat([new_seed_data, new_seed_sense], axis=1, ignore_index=True)
        new_seed_df.rename(columns={0: 'context', 1: 'sense'}, inplace=True)

        # Remove selected rows and reset the index
        data_df = data_df.drop(index=high_confidence_indices).reset_index(drop=True)
        final_small_annotated_data = pd.concat([final_small_annotated_data, new_seed_df], axis=0, ignore_index=True)

        X_dev = vectorizer.transform(dev_filtered_df['context'])
        predictions = classifier.predict(X_dev)
        dev_filtered_df = dev_filtered_df.copy(deep=True)
        dev_filtered_df['predictions'] = predictions
        true_dev_pred = len(dev_filtered_df[dev_filtered_df['actual_value']==dev_filtered_df['predictions']])
        dev_accuracy = true_dev_pred/len(dev_filtered_df['actual_value'])
        if data_df.shape[0] == 0:
            break
    accuracy.append(dev_accuracy)
    # # Step 4: Use the last model as the final model
    if test:
        final_model = classifier
        X_test = vectorizer.transform(test_filtered_df['context'])
        predictions = final_model.predict(X_test)
        test_filtered_df = test_filtered_df.copy(deep=True)
        test_filtered_df['predictions'] = predictions
        test_filtered_df.loc[:, 'actual_value'] = test_filtered_df['actual_value'].apply(lambda x: x[0])
        true_test_pred = len(test_filtered_df[test_filtered_df['actual_value'] == test_filtered_df['predictions']])
        test_accuracy = true_test_pred / len(test_filtered_df['actual_value'])
        accuracy.append(test_accuracy)
    return accuracy


def bootstrapping(dev_key, dev_instances, test_key, test_instances, iterations_number, word, confidence, preprocess_data=False, test=False):
    """
    Bootstrapping algorithm method for words deal, time, power, level, and sense.
    """
    # Convert class instances to dictionary
    dev_data_dict = {
        'id': [instance.id for key, instance in list(dev_instances.items())],
        'lemma': [instance.lemma for key, instance in list(dev_instances.items())],
        'context': [' '.join(instance.context) for key, instance in list(dev_instances.items())],
        'index': [instance.index for key, instance in list(dev_instances.items())],
        'pos': [instance.pos for key, instance in list(dev_instances.items())],
    }
    true_dev_instances_df = pd.DataFrame(dev_data_dict)
    cols = ['id', 'actual_value']
    dev_key_df = pd.DataFrame(list(dev_key.items()), columns=cols)

    dev_instances_df = pd.merge(true_dev_instances_df, dev_key_df, on="id", how='inner')
    test_data_dict = {
        'id': [instance.id for key, instance in list(test_instances.items())],
        'lemma': [instance.lemma for key, instance in list(test_instances.items())],
        'context': [' '.join(instance.context) for key, instance in list(test_instances.items())],
        'index': [instance.index for key, instance in list(test_instances.items())],
        'pos': [instance.pos for key, instance in list(test_instances.items())],
    }
    true_test_instances_df = pd.DataFrame(test_data_dict)
    test_key_df = pd.DataFrame(list(test_key.items()), columns=cols)
    test_instances_df = pd.merge(true_test_instances_df, test_key_df, on="id", how='inner')
    # masc data load for words time, power, level, sense
    # word sense
    if word == "sense":
        final_small_annotated_data, data_df = load_word_data('data/final_sense.csv', 'sense%1:09:05::', 'sense%1:09:04::', preprocess_data)
        dev_filtered_df = dev_instances_df[dev_instances_df['lemma'] == 'common_sense']
        test_filtered_df = test_instances_df[test_instances_df['lemma'] == 'sense']
        iterations_number = iterations_number
        dev_filtered_df.reset_index(inplace=True, drop='index')
        dev_filtered_df = dev_filtered_df.copy()
        dev_filtered_df = dev_filtered_df[['context', 'actual_value']]
        dev_filtered_df['actual_value'] = dev_filtered_df['actual_value'].apply(
            lambda x: [item.replace("common_sense%1:09:00::",
                                    "sense%1:09:04::") for item in x])
        dev_filtered_sense_1_df = data_df[data_df['sense'] == "sense%1:09:04::"].iloc[:5].rename(columns={'sense':'actual_value'})
        dev_filtered_sense_2_df = data_df[data_df['sense'] == "sense%1:09:05::"].iloc[:5].rename(columns={'sense':'actual_value'})
        dev_filtered_subset_df = pd.concat([dev_filtered_df, dev_filtered_sense_1_df, dev_filtered_sense_2_df], axis=0, ignore_index=True)
    if word == "time":
        final_small_annotated_data, data_df = load_word_data('data/final_time.csv', 'time%1:28:05::', 'time%1:11:00::', preprocess_data)
        dev_filtered_df = dev_instances_df[dev_instances_df['lemma'] == 'time']
        test_filtered_df = test_instances_df[test_instances_df['lemma'] == 'time']
        iterations_number = iterations_number
        dev_filtered_df.reset_index(inplace=True, drop='index')
        dev_filtered_df = dev_filtered_df.copy()
        dev_filtered_df = dev_filtered_df[['context', 'actual_value']]
        dev_filtered_sense_1_df = data_df[data_df['sense'] == "time%1:28:05::"].iloc[:5].rename(columns={'sense':'actual_value'})
        dev_filtered_sense_2_df = data_df[data_df['sense'] == "time%1:11:00::"].iloc[:5].rename(columns={'sense':'actual_value'})
        dev_filtered_subset_df = pd.concat([dev_filtered_df, dev_filtered_sense_1_df, dev_filtered_sense_2_df], axis=0, ignore_index=True)
    if word == "level":
        final_small_annotated_data, data_df = load_word_data('data/final_level.csv', 'level%1:26:01::', 'level%1:07:00::', preprocess_data)
        dev_filtered_df = dev_instances_df[dev_instances_df['lemma'] == 'level']
        test_filtered_df = test_instances_df[test_instances_df['lemma'] == 'level']
        iterations_number = iterations_number
        dev_filtered_df.reset_index(inplace=True, drop='index')
        dev_filtered_df = dev_filtered_df.copy()
        dev_filtered_df = dev_filtered_df[['context', 'actual_value']]
        dev_filtered_sense_1_df = data_df[data_df['sense'] == "level%1:26:01::"].iloc[:5].rename(columns={'sense':'actual_value'})
        dev_filtered_sense_2_df = data_df[data_df['sense'] == "level%1:07:00::"].iloc[:5].rename(columns={'sense':'actual_value'})
        dev_filtered_subset_df = pd.concat([dev_filtered_df, dev_filtered_sense_1_df, dev_filtered_sense_2_df], axis=0, ignore_index=True)
    if word == "power":
        final_small_annotated_data, data_df = load_word_data('final_power.csv', 'power%1:07:02::', 'power%1:14:00::', preprocess_data)
        dev_filtered_df = dev_instances_df[dev_instances_df['lemma'] == 'major_power']
        test_filtered_df = test_instances_df[test_instances_df['lemma'] == 'power']
        iterations_number = iterations_number
        dev_filtered_df.reset_index(inplace=True, drop='index')
        dev_filtered_df = dev_filtered_df.copy()
        dev_filtered_df = dev_filtered_df[['context', 'actual_value']]
        dev_filtered_df['actual_value'] = dev_filtered_df['actual_value'].apply(
            lambda x: [item.replace("major_power%1:14:00::",
                                    "power%1:14:00::") for item in x])
        dev_filtered_sense_1_df = data_df[data_df['sense'] == "power%1:07:02::"].iloc[:5].rename(columns={'sense':'actual_value'})
        dev_filtered_sense_2_df = data_df[data_df['sense'] == "power%1:14:00::"].iloc[:5].rename(columns={'sense':'actual_value'})
        dev_filtered_subset_df = pd.concat([dev_filtered_df, dev_filtered_sense_1_df, dev_filtered_sense_2_df], axis=0, ignore_index=True)
    if word == "deal":
        final_small_annotated_data, data_df = load_word_data('data/final_deal.csv', 'deal%1:04:02::', 'deal%1:10:00::', preprocess_data)
        dev_filtered_df = dev_instances_df[dev_instances_df['lemma'] == 'deal']
        test_filtered_df = test_instances_df[test_instances_df['lemma'] == 'deal']
        iterations_number = iterations_number
        dev_filtered_df.reset_index(inplace=True, drop='index')
        dev_filtered_df = dev_filtered_df.copy()
        dev_filtered_df = dev_filtered_df[['context', 'actual_value']]
        dev_filtered_sense_1_df = data_df[data_df['sense'] == "deal%1:04:02::"].iloc[:5].rename(columns={'sense':'actual_value'})
        # dev_filtered_sense_2_df = data_df[data_df['sense'] == "deal%1:10:00::"].iloc[:5].rename(columns={'sense':'actual_value'})
        dev_filtered_subset_df = pd.concat([dev_filtered_df, dev_filtered_sense_1_df], axis=0, ignore_index=True)
    if word == "life":
        final_small_annotated_data, data_df = load_word_data('data/final_life.csv', 'life%1:19:00::', 'life%1:26:00::', preprocess_data)
        dev_filtered_df = dev_instances_df[dev_instances_df['lemma'] == 'life']
        test_filtered_df = test_instances_df[test_instances_df['lemma'] == 'life']
        iterations_number = iterations_number
        dev_filtered_df.reset_index(inplace=True, drop='index')
        dev_filtered_df = dev_filtered_df.copy()
        dev_filtered_df = dev_filtered_df[['context', 'actual_value']]
        dev_filtered_sense_1_df = data_df[data_df['sense'] == "life%1:19:00::"].iloc[:5].rename(columns={'sense': 'actual_value'})
        dev_filtered_sense_2_df = data_df[data_df['sense'] == "life%1:26:00::"].iloc[:5].rename(columns={'sense': 'actual_value'})
        dev_filtered_subset_df = pd.concat([dev_filtered_df, dev_filtered_sense_1_df, dev_filtered_sense_2_df], axis=0, ignore_index=True)
    dev_filtered_subset_df['actual_value'] = dev_filtered_subset_df['actual_value'].apply(lambda x: x[0] if isinstance(x, list) else x)
    dev_filtered_df = dev_filtered_subset_df
    data_df = pd.concat([data_df, dev_filtered_sense_1_df.rename(columns={'actual_value': 'sense'})]).drop_duplicates(keep=False)

    data_df.reset_index(inplace=True, drop='index')
    if not test:
        bootstrapping_accuracy = check_model_accuracy(final_small_annotated_data, data_df, dev_filtered_df, test_filtered_df, iterations_number, confidence, preprocess_data)
    else:
        bootstrapping_accuracy = check_model_accuracy(final_small_annotated_data, data_df, dev_filtered_df, test_filtered_df, iterations_number, confidence, preprocess_data, True)
    return bootstrapping_accuracy
