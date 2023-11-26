from nltk.tokenize import word_tokenize
from collections import Counter
import os
import numpy as np
import csv
import re
from tqdm import tqdm

from utils import paths, fs

years = list(range(2016, 2022 + 1))

Vocabulary = fs.load_json_file(f"{paths.vocabulary_word_dir}/Whole_Vocabulary.json")
word_counts_total = fs.load_json_file(f"{paths.vocabulary_word_counts_dir}/word_counts_total.json")
result_csv_headers = ["FILE_NAME", "CIK", "YEAR"] + Vocabulary


def calculate_TF(file_path):
    global Vocabulary

    with open(file_path, 'r') as file:
        file_content = file.read()

    file_content = file_content.replace("\n", " ")
    word_list = word_tokenize(file_content)

    word_counts = Counter(word_list)
    TF_Dic = {word: word_counts[word] if word in word_counts else 0 for word in Vocabulary}
    return TF_Dic


def get_the_number_of_documents(folder_path):
    try:
        files = [file_name for file_name in os.listdir(folder_path) if file_name.endswith(".htm.txt")]
        number_of_documents = len(files)

        return number_of_documents
    except FileNotFoundError:
        print(f"Folder not found: {folder_path}")
        return None


number_of_documents_by_year = {}
for year in range(2016, 2023):
    year = str(year)
    number_of_documents = get_the_number_of_documents(f'{paths.train_dir}/{year}')
    number_of_documents_by_year[year] = number_of_documents

number_of_documents_in_total = sum(number_of_documents_by_year.values())


def calculate_IDF(word_counts_total, number_of_documents_in_total):
    global Vocabulary
    IDF_Dic = {key: np.log(number_of_documents_in_total / value)
               for key, value in word_counts_total.items()}

    missing_words = [word for word in Vocabulary if word not in IDF_Dic]

    if len(missing_words) != 0:
        error_message = f"Error: Missing words in the IDF output: {missing_words}"
        return error_message

    else:
        return IDF_Dic


IDF = calculate_IDF(word_counts_total, number_of_documents_in_total)


# TFIDF Matrix
def calculate_TFIDF(file_path, IDF):
    TF = calculate_TF(file_path)

    TFIDF = {key: TF[key] * IDF[key] for key in TF}

    return TFIDF


def get_cik(file_name):
    match = re.search(r'^(\d+)_', file_name)
    cik = int(match.group(1))
    return cik


def save_TFIDF_to_csv(folder_path, IDF):
    year = folder_path[-4:]
    file_list = os.listdir(folder_path)

    text_files = [file for file in file_list if file.endswith(".txt")]
    # text_files = sorted(text_files, key=lambda filename: int(filename.split("_")[0]))

    with open(f"{paths.tfidf_dir}/{year}_TFIDF.csv", "w") as fp:
        writer = csv.DictWriter(fp, fieldnames=result_csv_headers)
        writer.writeheader()

        with tqdm(total=len(text_files), unit="file", desc=f"Get TFIDF of {year}") as pbar_tfidf:
            for file_name in text_files:
                file_path = os.path.join(folder_path, file_name)

                TFIDF_dict = calculate_TFIDF(file_path, IDF)

                Company_year_TFIDF = {
                    "FILE_NAME": file_name,
                    "CIK": file_name.split("_")[0],
                    "YEAR": year,
                    **TFIDF_dict
                }

                # Add new TFIDF_dict to TFIDF csv File
                writer.writerow(Company_year_TFIDF)
                pbar_tfidf.update(1)


def get_TFIDF(folder_path, IDF):
    year = folder_path[-4:]
    file_list = os.listdir(folder_path)

    text_files = [file for file in file_list if file.endswith(".txt")]

    for file_name in text_files:
        file_path = os.path.join(folder_path, file_name)

        TFIDF_dict = calculate_TFIDF(file_path, IDF)

        Company_year_info = {}
        Company_year_info["FILE_NAME"] = file_name
        cik = get_cik(file_name)
        Company_year_info["CIK"] = cik
        Company_year_info["YEAR"] = year

        Company_year_TFIDF = {**Company_year_info, **TFIDF_dict}

        TFIDF_path = f'{paths.tfidf_dir}/{year}_TFIDF.json'

        # Add new TFIDF_dict to TFIDF Json File
        TFIDF = fs.load_json_file(TFIDF_path)
        TFIDF.append(Company_year_TFIDF)
        fs.save_json_file(TFIDF_path, TFIDF)


fs.clear_dir(paths.tfidf_dir)

folder_path_list = [f"{paths.train_dir}/{year}" for year in years]
for folder_path in folder_path_list:
    # get_TFIDF(folder_path, IDF)  # JSON too slow
    save_TFIDF_to_csv(folder_path, IDF)
