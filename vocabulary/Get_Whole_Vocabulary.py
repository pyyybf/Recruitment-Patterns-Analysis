#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from nltk.tokenize import word_tokenize
import json
import ast


# ## Get the number of documents

# ## Get the vocabulary of year and word counts by year

# ## Get the whole vocabulary and word counts in total

# In[2]:


def read_vocabulary_sets(file_path):
    with open(file_path, 'r') as file:
        vocabulary_sets = file.read()
    
    vocabulary_sets = ast.literal_eval(vocabulary_sets)
    return vocabulary_sets 


# In[3]:


def get_whole_year_vocabulary(file_path):  
    
    vocabulary_sets = read_vocabulary_sets(file_path)
    
    Vocabulary = set()
    
    for i in vocabulary_sets:
        Vocabulary.update(i)
    
    return Vocabulary


# In[4]:


def get_whole_year_vocabulary_word_counts(file_path):
    
    vocabulary_sets = read_vocabulary_sets(file_path)
    
    Vocabulary = get_whole_year_vocabulary(file_path)
    
    word_counts = {target_word: sum(sublist.count(target_word) for sublist in vocabulary_sets) 
                   for target_word in Vocabulary}

    return word_counts


# In[5]:


def save_file(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file)


# In[6]:


word_counts_by_year = {}
Whole_Vocabulary = set()

for year in range(2016, 2022):
    year = str(year)
    Year_Vocabulary = get_whole_year_vocabulary(f'vocabulary_sets/{year}_vocabulary_sets.json')
    
    Whole_Vocabulary = Whole_Vocabulary.union(Year_Vocabulary)
    
    word_counts = get_whole_year_vocabulary_word_counts(f'vocabulary_sets/{year}_vocabulary_sets.json')
    for word, count in word_counts.items():
        word_counts_by_year.setdefault(word, {}).update({year: count})
    
    Year_Vocabulary = list(Year_Vocabulary)
    save_file(f'Vocabulary/word/{year}_Vocabulary.json', Year_Vocabulary)
    save_file(f'Vocabulary/word_counts/{year}_word_counts.json', word_counts)
    
word_counts_total = {word: sum(counts.values()) for word, counts in word_counts_by_year.items()}


# In[7]:


Whole_Vocabulary = list(Whole_Vocabulary)
save_file('Vocabulary/word/Whole_Vocabulary.json', Whole_Vocabulary)


# In[8]:


save_file('Vocabulary/word_counts/word_counts_total.json', word_counts_total)


# In[ ]:





# In[ ]:




