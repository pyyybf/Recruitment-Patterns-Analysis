#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
from nltk.tokenize import word_tokenize
from collections import Counter
import os
import numpy as np
import csv
import re


# In[2]:


def load_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = []
    return data


# In[3]:


def save_file(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file)


# In[4]:


Vocabulary = load_file('Vocabulary/word/Whole_Vocabulary.json')


# In[5]:


word_counts_total = load_file('Vocabulary/word_counts/word_counts_total.json')


# In[6]:


def calculate_TF(file_path):
    global Vocabulary

    with open(file_path, 'r') as file:
        file_content = file.read()
    
    file_content = file_content.replace("\n", " ")
    word_list = word_tokenize(file_content)
    
    word_counts = Counter(word_list)
    TF_Dic = {word: word_counts[word] if word in word_counts else 0 for word in Vocabulary}
    return TF_Dic


# In[7]:


def get_the_number_of_documents(folder_path):
    try:
        files = os.listdir(folder_path)

        number_of_documents = len(files)

        return number_of_documents
    except FileNotFoundError:
        print(f"Folder not found: {folder_path}")
        return None


# In[8]:


number_of_documents_by_year = {}
for year in range(2016, 2022):
    year = str(year)
    number_of_documents = get_the_number_of_documents(f'data/{year}')

    number_of_documents_by_year[year] = number_of_documents

number_of_documents_in_total = sum(number_of_documents_by_year.values())


# In[9]:


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


# In[10]:


IDF = calculate_IDF(word_counts_total, number_of_documents_in_total)


# ## TFIDF Matrix

# In[11]:


def calculate_TFIDF(file_path, IDF):
    
    TF = calculate_TF(file_path)
    
    TFIDF = {key: TF[key] * IDF[key] for key in TF}
    
    return TFIDF


# In[12]:


def get_TFIDF(folder_path, IDF):
    year = folder_path[-4:]
    file_list = os.listdir(folder_path)
    
    text_files = [file for file in file_list if file.endswith(".txt")]
    
    for file_name in text_files:
    
        file_path = os.path.join(folder_path, file_name)
        
        TFIDF_dict = calculate_TFIDF(file_path, IDF)
        
        Company_year_info = {}
        Company_year_info["FILE_NAME"] = file_name
        match = re.search(r'^(\d+)_', file_name)
        cik = int(match.group(1))
        Company_year_info["CIK"] = cik
        Company_year_info["YEAR"] = year
        
        Company_year_TFIDF = {**Company_year_info, **TFIDF_dict}
        
        TFIDF_path = 'Results/TFIDF/TFIDF.json'
        
        # Add new TFIDF_dict to TFIDF Json File
        TFIDF = load_file(TFIDF_path)
        TFIDF.append(Company_year_TFIDF)
        save_file(TFIDF_path, TFIDF)


# In[13]:


folder_path_list = ["data/2016", "data/2017", "data/2018", "data/2019", "data/2020", "data/2021"]
for folder_path in folder_path_list:
    get_TFIDF(folder_path, IDF)


# In[14]:


def json_to_csv(csv_file_name, json_file):
    with open(csv_file_name, 'w', newline='') as csv_file:
        fieldnames = json_file[0].keys()
        
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        writer.writeheader()
        
        writer.writerows(json_file)
        


# In[15]:


TFIDF = load_file('Results/TFIDF/TFIDF.json')
json_to_csv('Results/TFIDF/TFIDF.csv', TFIDF)


# In[ ]:




