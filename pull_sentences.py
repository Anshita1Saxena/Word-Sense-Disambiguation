import nltk

nltk.download('reuters')
from nltk.corpus import reuters
from nltk import sent_tokenize, word_tokenize


# Function to check if a sentence contains the exact word "action"
def contains_exact_action(sentence):
    words = word_tokenize(sentence)
    return 'deal' in words


# Get a list of fileids in the 'reuters' corpus
file_ids = reuters.fileids()

# Initialize an empty list to store sentences
exact_action_sentences = []

# Iterate over the file ids and extract sentences
for file_id in file_ids:
    # Load the raw text of the document
    raw_text = reuters.raw(file_id)

    # Tokenize the raw text into sentences
    doc_sentences = sent_tokenize(raw_text)

    # Filter sentences containing the exact word "action"
    filtered_sentences = filter(contains_exact_action, doc_sentences)

    # Add the filtered sentences to the list
    exact_action_sentences.extend(filtered_sentences)

    # Break the loop when we have 110 sentences
    if len(exact_action_sentences) >= 300:
        break

write_file = open("deal.txt", "w")
# Print the first 110 sentences containing the exact word "action"
for i, sentence in enumerate(exact_action_sentences[:300]):
    print(f"{i + 1}. {sentence}")
    write_file.write(sentence)
    write_file.write("\n")
write_file.close()
