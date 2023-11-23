#!/usr/bin/env python
# coding: utf-8

# In[1]:

import json
import os
from nltk.tokenize import word_tokenize
import re
import csv
import shutil
from tqdm import tqdm

# In[2]:

data_base_dir = "./data_split/train_data"
years = list(range(2016, 2022 + 1))
whole_vocabulary_json_path = './word/Whole_Vocabulary.json'
result_dir = './Incidence_Matrix/csv_file'

def load_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = []
    return data


# In[3]:


Vocabulary = load_json_file(whole_vocabulary_json_path)
result_csv_headers = ["FILE_NAME", "CIK", "YEAR"] + Vocabulary


# In[4]:


def save_json_file(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file)


# In[5]:


def extract_vocabulary(file_content):
    file_content = file_content.replace("\n", " ")
    word_list = word_tokenize(file_content)
    vocabulary = set(word_list)

    return vocabulary


# In[6]:


def get_Incidence(file_path, vocabulary_dict):
    
    with open(file_path, 'r') as file:
        file_content = file.read()
        
    words_in_text = extract_vocabulary(file_content)
    
    Incidence_dict = {word: 1 if word in words_in_text else 0 
                      for word in vocabulary_dict}
    
    return Incidence_dict


# In[7]:
def save_Incidence_Matrix_to_csv(folder_path, vocabulary_dict):
    year = folder_path[-4:]
    file_list = os.listdir(folder_path)

    text_files = [file for file in file_list if file.endswith(".txt")]

    with open(f"{result_dir}/{year}_Incidence_Matrix.csv", "w") as fp:
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


# In[8]:
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
        
        Incidence_Matrix_path = f'{result_dir}/{year}_Incidence_Matrix.json'
        
        # Add new TFIDF_dict to TFIDF Json File
        Incidence_Matrix = load_json_file(Incidence_Matrix_path)
        Incidence_Matrix.append(Company_year_Incidence)
        save_json_file(Incidence_Matrix_path, Incidence_Matrix)


# In[9]:

def clear_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

clear_dir(result_dir)


# In[10]:

folder_path_list = [f"{data_base_dir}/{year}" for year in years]
for folder_path in folder_path_list:
    save_Incidence_Matrix_to_csv(folder_path, Vocabulary)
