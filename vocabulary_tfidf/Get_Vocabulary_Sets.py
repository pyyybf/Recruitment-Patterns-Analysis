#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from nltk.tokenize import word_tokenize
import json
import ast
import nltk
from nltk.corpus import stopwords
import shutil


# ## Get and save vocabulary_sets

# In[2]:


def filter_words(words):
    stop_words = set(stopwords.words('english'))
    html_stop_words = ['nbsp', 'quot', 'amp', 'lt', 'gt', 'apos', 'middot', 
                       'ldquo', 'rdquo', 'lsquo', 'rsquo', 'sbquo', 'ndash', 
                       'mdash', 'hellip', 'bull', 'pr', 'laquo', 'raquo']
    stop_words.update(html_stop_words)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return filtered_words


# In[3]:


def extract_vocabulary(file_content):
    file_content = file_content.replace("\n", " ")
    words = word_tokenize(file_content)
    words = filter_words(words)
    vocabulary = set(words)
    return vocabulary


# In[4]:


def load_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = []
    return data


# In[5]:


def save_json_file(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file)


# In[6]:


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
        
        vocabulary_path = f'vocabulary_sets/{year}_vocabulary_sets.json'
        
        # Add new vocabulary set to Vocabulary Json File
        Vocabulary = load_json_file(vocabulary_path)
        Vocabulary.append(list(new_vocabulary_set))
        save_json_file(vocabulary_path, Vocabulary)


# In[7]:


def clear_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


# In[8]:


clear_dir('vocabulary_sets')


# In[9]:


folder_path_list = ["data_split/train_data/2016", "data_split/train_data/2017", 
                    "data_split/train_data/2018", "data_split/train_data/2019", 
                    "data_split/train_data/2020", "data_split/train_data/2021", 
                    "data_split/train_data/2022"]
for folder_path in folder_path_list:
    get_vocabulary_sets(folder_path)


# In[ ]:




