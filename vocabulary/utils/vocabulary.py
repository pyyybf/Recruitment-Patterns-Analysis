from nltk.tokenize import word_tokenize
import json

def extract_vocabulary(file_content):
    file_content = file_content.replace("\n", " ")
    word_list = word_tokenize(file_content)
    new_vocabulary_set = set(word_list)
    return new_vocabulary_set

def load_sets_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            sets_data = json.load(file)
    except FileNotFoundError:
        sets_data = []
    return sets_data

def save_sets_to_file(file_path, sets_data):
    with open(file_path, 'w') as file:
        json.dump(sets_data, file)
