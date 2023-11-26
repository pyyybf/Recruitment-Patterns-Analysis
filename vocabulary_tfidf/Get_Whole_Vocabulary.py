#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from nltk.tokenize import word_tokenize
import json
import ast
import shutil


# ## Get the number of documents

# ## Get the vocabulary of year and word counts by year

# ## Get the whole vocabulary and word counts in total

# In[2]:


def load_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = []
    return data


# In[3]:


def read_vocabulary_sets(file_path):
    with open(file_path, 'r') as file:
        vocabulary_sets = file.read()
    
    vocabulary_sets = ast.literal_eval(vocabulary_sets)
    return vocabulary_sets 


# In[4]:


def get_whole_year_vocabulary(file_path):  
    
    vocabulary_sets = read_vocabulary_sets(file_path)
    
    Vocabulary = set()
    
    for i in vocabulary_sets:
        Vocabulary.update(i)
    
    return Vocabulary


# In[5]:


def get_whole_year_vocabulary_word_counts(file_path):
    
    vocabulary_sets = read_vocabulary_sets(file_path)
    
    Vocabulary = get_whole_year_vocabulary(file_path)
    
    word_counts = {target_word: sum(sublist.count(target_word) for sublist in vocabulary_sets) 
                   for target_word in Vocabulary}

    return word_counts


# In[6]:


def save_json_file(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file)


# In[7]:


def clear_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


# In[8]:


clear_dir('Vocabulary/word')
clear_dir('Vocabulary/word_counts')


# In[9]:


word_counts_total = {}
Whole_Vocabulary = set()

for year in range(2016, 2023):
    year = str(year)
    Year_Vocabulary = get_whole_year_vocabulary(f'vocabulary_sets/{year}_vocabulary_sets.json')
    
    Whole_Vocabulary = Whole_Vocabulary.union(Year_Vocabulary)
    
    word_counts_year = get_whole_year_vocabulary_word_counts(f'vocabulary_sets/{year}_vocabulary_sets.json')

    Year_Vocabulary = list(Year_Vocabulary)
    save_json_file(f'Vocabulary/word/{year}_Vocabulary.json', Year_Vocabulary)
    save_json_file(f'Vocabulary/word_counts/{year}_word_counts.json', word_counts_year)
    
    if year=='2016':
        word_counts_total = word_counts_year
    else:
        for key in set(word_counts_total.keys()) | set(word_counts_year.keys()):
            word_counts_total[key] = word_counts_total.get(key, 0) + word_counts_year.get(key, 0)


# In[10]:


Whole_Vocabulary = list(Whole_Vocabulary)
save_json_file('Vocabulary/word/Whole_Vocabulary.json', Whole_Vocabulary)


# In[11]:


save_json_file('Vocabulary/word_counts/word_counts_total.json', word_counts_total)


# In[ ]:




