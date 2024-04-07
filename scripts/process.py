import json
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from word2number import w2n
import torch



TREEBANK_TO_WORDNET_POS = {
    'J': wordnet.ADJ,
    'V': wordnet.VERB,
    'N': wordnet.NOUN,
    'R': wordnet.ADV
}


def is_missing_attr(json_review):
    # Finding missing attribute
    important_attr = ['stars', 'useful', 'funny', 'cool', 'text']
    for attr in important_attr:
        if attr not in json_review:
            print(f"{attr} does not exist in the json line\n{json_review}")
            return False
        return True


def clean_punctuation(text_val):
    # remove special characters (?,!,#,-,/ ...)
    cleaned = re.sub(r'[^\w\s]|["\r\n]', '', text_val)
    return cleaned


def clean_stopwords(text_val):
    # https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text_val)
    filtered_val = [w for w in word_tokens if not w.lower() in stop_words]

    for w in word_tokens:
        if w not in stop_words:
            filtered_val.append(w)

    return ' '.join(filtered_val)


def clean_numbers(text_val):
    text_list = text_val.split()
    text_len = len(text_list)
    # Convert number words to numbers - e.g., two -> 2
    for i in range(text_len):
        try:
            text_list[i] = str(w2n.word_to_num(text_list[i]))
        except:
            pass

    # Removing number digits from the text
    new_text_list = []
    for word in text_list:
        if not word.isdigit():
            new_text_list.append(word)

    cleaned = ' '.join(new_text_list)

    return cleaned


def get_wordnet_pos(treebank_tag):
    """Convert the part-of-speech naming scheme
    from the Penn Treebank tag to a WordNet tag"""
    # By default, WordNetLemmatizer assumes everything is a noun
    return TREEBANK_TO_WORDNET_POS.get(treebank_tag[0], wordnet.NOUN)


def clean_lemmatization(text_val):
    lemmatizer = WordNetLemmatizer()
    # Tokenize the sentence
    word_pos_tags = pos_tag(text_val.split())
    # Convert to a list of tuples with word and corresponding WordNet POS tag
    new_text = [lemmatizer.lemmatize(word, get_wordnet_pos(pos_tag)) for
                word, pos_tag in word_pos_tags]

    return ' '.join(new_text)


def process_line(json_line):
    new_text = json_line['text'].lower()
    new_text = clean_punctuation(new_text)
    # new_text = clean_stopwords(new_text)
    # new_text = clean_lemmatization(new_text)
    # new_text = clean_numbers(new_text)
    json_line['text'] = new_text
    return json_line


def clean_json(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8', errors='ignore') as file:
        data = []
        for line in file:
            json_line = json.loads(line)

            processed_line = process_line(json_line)
            data.append(processed_line)
    save_to_csv(data, output_file_path)


def save_to_csv(data, file_path):
    reviews = pd.DataFrame(data, columns=['stars', 'useful', 'funny', 'cool',
                                          'text'])
    reviews = reviews.loc[(reviews['funny'] != -1) &
                          (reviews['useful'] != -1) &
                          (reviews['cool'] != -1)]
    reviews.to_csv(file_path, mode='a', index=False,
                   header=not pd.io.common.file_exists(file_path))

