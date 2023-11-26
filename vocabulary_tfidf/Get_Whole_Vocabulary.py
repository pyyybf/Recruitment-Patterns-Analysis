import ast

from utils import paths, fs


# Get the number of documents
# Get the vocabulary of year and word counts by year
# Get the whole vocabulary and word counts in total
def read_vocabulary_sets(file_path):
    with open(file_path, 'r') as file:
        vocabulary_sets = file.read()

    vocabulary_sets = ast.literal_eval(vocabulary_sets)
    return vocabulary_sets


def get_whole_year_vocabulary(file_path):
    vocabulary_sets = read_vocabulary_sets(file_path)

    Vocabulary = set()

    for i in vocabulary_sets:
        Vocabulary.update(i)

    return Vocabulary


def get_whole_year_vocabulary_word_counts(file_path):
    vocabulary_sets = read_vocabulary_sets(file_path)

    Vocabulary = get_whole_year_vocabulary(file_path)

    word_counts = {target_word: sum(sublist.count(target_word) for sublist in vocabulary_sets)
                   for target_word in Vocabulary}

    return word_counts


fs.clear_dir(paths.vocabulary_word_dir)
fs.clear_dir(paths.vocabulary_word_counts_dir)

word_counts_total = {}
Whole_Vocabulary = set()

for year in range(2016, 2023):
    year = str(year)
    Year_Vocabulary = get_whole_year_vocabulary(f'{paths.vocabulary_sets_dir}/{year}_vocabulary_sets.json')

    Whole_Vocabulary = Whole_Vocabulary.union(Year_Vocabulary)

    word_counts_year = get_whole_year_vocabulary_word_counts(f'{paths.vocabulary_sets_dir}/{year}_vocabulary_sets.json')

    Year_Vocabulary = list(Year_Vocabulary)
    fs.save_json_file(f'{paths.vocabulary_word_dir}/{year}_Vocabulary.json', Year_Vocabulary)
    fs.save_json_file(f'{paths.vocabulary_word_counts_dir}/{year}_word_counts.json', word_counts_year)

    if year == '2016':
        word_counts_total = word_counts_year
    else:
        for key in set(word_counts_total.keys()) | set(word_counts_year.keys()):
            word_counts_total[key] = word_counts_total.get(key, 0) + word_counts_year.get(key, 0)

Whole_Vocabulary = list(Whole_Vocabulary)
fs.save_json_file(f'{paths.vocabulary_word_dir}/Whole_Vocabulary.json', Whole_Vocabulary)
fs.save_json_file(f'{paths.vocabulary_word_counts_dir}/word_counts_total.json', word_counts_total)
