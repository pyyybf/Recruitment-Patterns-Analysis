import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from utils import paths, fs


# Get and save vocabulary_sets
def filter_words(words):
    stop_words = set(stopwords.words('english'))
    html_stop_words = ['nbsp', 'quot', 'amp', 'lt', 'gt', 'apos', 'middot',
                       'ldquo', 'rdquo', 'lsquo', 'rsquo', 'sbquo', 'ndash',
                       'mdash', 'hellip', 'bull', 'pr', 'laquo', 'raquo']
    stop_words.update(html_stop_words)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return filtered_words


def extract_vocabulary(file_content):
    file_content = file_content.replace("\n", " ")
    words = word_tokenize(file_content)
    words = filter_words(words)
    vocabulary = set(words)
    return vocabulary


def get_vocabulary_sets(folder_path):
    year = folder_path[-4:]
    # List all files in the folder
    file_list = os.listdir(folder_path)

    # Filter only text files if needed
    text_files = [file for file in file_list if file.endswith(".txt")]

    for file_name in text_files:
        file_path = os.path.join(folder_path, file_name)

        with open(file_path, 'r') as file:
            file_content = file.read()
        new_vocabulary_set = extract_vocabulary(file_content)

        vocabulary_path = f'{paths.vocabulary_sets_dir}/{year}_vocabulary_sets.json'

        # Add new vocabulary set to Vocabulary Json File
        Vocabulary = fs.load_json_file(vocabulary_path)
        Vocabulary.append(list(new_vocabulary_set))
        fs.save_json_file(vocabulary_path, Vocabulary)


fs.clear_dir(paths.vocabulary_sets_dir)

for year in range(2016, 2023):
    get_vocabulary_sets(f"{paths.base_dir}/{year}")
