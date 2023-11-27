import os
from nltk.tokenize import word_tokenize
import re
import csv
from tqdm import tqdm

from utils import paths, fs

years = list(range(2016, 2022 + 1))

Vocabulary = fs.load_json_file(f"{paths.vocabulary_word_dir}/Whole_Vocabulary.json")
result_csv_headers = ["FILE_NAME", "CIK", "YEAR"] + Vocabulary


def extract_vocabulary(file_content):
    file_content = file_content.replace("\n", " ")
    word_list = word_tokenize(file_content)
    vocabulary = set(word_list)

    return vocabulary


def get_Incidence(file_path, vocabulary_dict):
    with open(file_path, 'r') as file:
        file_content = file.read()

    words_in_text = extract_vocabulary(file_content)

    Incidence_dict = {word: 1 if word in words_in_text else 0
                      for word in vocabulary_dict}

    return Incidence_dict


def save_Incidence_Matrix_to_csv(folder_path, vocabulary_dict):
    year = folder_path[-4:]
    file_list = os.listdir(folder_path)

    text_files = [file for file in file_list if file.endswith(".txt")]

    with open(f"{paths.incmat_dir}/{year}_Incidence_Matrix.csv", "w") as fp:
        writer = csv.DictWriter(fp, fieldnames=result_csv_headers)
        writer.writeheader()

        with tqdm(total=len(text_files), unit="file", desc=f"Get Incidence_Matrix of {year}") as pbar_tfidf:
            for file_name in text_files:
                file_path = os.path.join(folder_path, file_name)

                Incidence_dict = get_Incidence(file_path, vocabulary_dict)

                Company_year_Incidence_Matrix = {
                    "FILE_NAME": file_name,
                    "CIK": file_name.split("_")[0],
                    "YEAR": year,
                    **Incidence_dict
                }

                # Add new Incidence_dict to Incidence_Matrix csv File
                writer.writerow(Company_year_Incidence_Matrix)
                pbar_tfidf.update(1)


def get_Incidence_Matrix(folder_path, vocabulary_dict):
    year = folder_path[-4:]
    file_list = os.listdir(folder_path)

    text_files = [file for file in file_list if file.endswith(".txt")]

    for file_name in text_files:
        file_path = os.path.join(folder_path, file_name)

        Incidence_dict = get_Incidence(file_path, vocabulary_dict)

        Company_year_info = {}
        Company_year_info["FILE_NAME"] = file_name
        match = re.search(r'^(\d+)_', file_name)
        cik = int(match.group(1))
        Company_year_info["CIK"] = cik
        Company_year_info["YEAR"] = year

        Company_year_Incidence = {**Company_year_info, **Incidence_dict}

        Incidence_Matrix_path = f'{paths.incmat_dir}/{year}_Incidence_Matrix.json'

        # Add new TFIDF_dict to TFIDF Json File
        Incidence_Matrix = fs.load_json_file(Incidence_Matrix_path)
        Incidence_Matrix.append(Company_year_Incidence)
        fs.save_json_file(Incidence_Matrix_path, Incidence_Matrix)


fs.clear_dir(paths.incmat_dir)

folder_path_list = [f"{paths.base_dir}/{year}" for year in years]
for folder_path in folder_path_list:
    save_Incidence_Matrix_to_csv(folder_path, Vocabulary)
