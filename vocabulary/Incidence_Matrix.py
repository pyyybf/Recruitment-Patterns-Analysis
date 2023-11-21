#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import os
from nltk.tokenize import word_tokenize
import re
import csv


# In[2]:


def load_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = []
    return data


# In[3]:


Vocabulary = load_file('Vocabulary/word/Whole_Vocabulary.json')


# In[4]:


def save_file(file_path, data):
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
        
        Incidence_Matrix_path = 'Results/Incidence_Matrix/Incidence_Matrix.json'
        
        # Add new TFIDF_dict to TFIDF Json File
        Incidence_Matrix = load_file(Incidence_Matrix_path)
        Incidence_Matrix.append(Company_year_Incidence)
        save_file(Incidence_Matrix_path, Incidence_Matrix)


# In[8]:


folder_path_list = ["data/2016", "data/2017", "data/2018", "data/2019", "data/2020", "data/2021"]
for folder_path in folder_path_list:
    get_Incidence_Matrix(folder_path, Vocabulary)


# In[9]:


def json_to_csv(csv_file_name, json_file):
    with open(csv_file_name, 'w', newline='') as csv_file:
        fieldnames = json_file[0].keys()
        
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        writer.writeheader()
        
        writer.writerows(json_file)
        


# In[10]:


Incidence_Matrix = load_file('Results/Incidence_Matrix/Incidence_Matrix.json')
json_to_csv('Results/Incidence_Matrix/Incidence_Matrix.csv', Incidence_Matrix)

