import numpy as np
import pandas as pd
import string
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
from nltk import pos_tag

# NLTK下载, 仅需下载一次
# nltk.download('omw-1.4')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')

# from const.stopwords import STOPWORDS

def processor_use_stem(texts, stopwords):
    result = []
    
    # 创建一个Porter词干提取器
    stemmer = PorterStemmer()
    
    # 初始化词形还原器
    # lemmatizer = WordNetLemmatizer()
    
    # 定义英文字符和空白字符的正则表达式模式
    english_chars_and_whitespace_pattern = re.compile(r'[^a-zA-Z\s]')
    
    # 定义英文单词的正则表达式模式
    english_word_pattern = re.compile(r'^[a-zA-Z-]+$')
    
    # 创建一个转换表，将所有标点符号映射为None
    translator = str.maketrans('', '', string.punctuation)
    
    for text in texts:
        cleaned_words = []
        for word in text.split():
            # 转换为小写
            word = word.lower()
            # 删除标点符号
            word = word.translate(translator)
            # 删除非英文字符
            word = english_chars_and_whitespace_pattern.sub('', word)
            
            if not word:
                continue
            
            # 提取词干
            word = stemmer.stem(word)
        
            # 进行词形还原
            # word = lemmatizer.lemmatize(word).strip()
            
            # 检查是否是英文单词，且不在停用词列表中
            # if english_word_pattern.search(word) and word not in stopwords:
            #     cleaned_words.append(word)
            
            if word and word not in stopwords:
                cleaned_words.append(word)
            
        # 转换为字符串
        cleaned_text = ' '.join(cleaned_words)
        result.append(cleaned_text)
    
    return result

def processor_use_lemma(texts, stopwords):
    result = []
    
    # 创建一个Porter词干提取器
    # stemmer = PorterStemmer()
    
    # 初始化词形还原器
    lemmatizer = WordNetLemmatizer()
    
    # 定义英文字符和空白字符的正则表达式模式
    english_chars_and_whitespace_pattern = re.compile(r'[^a-zA-Z\s]')
    
    # 创建一个转换表，将所有标点符号映射为None
    translator = str.maketrans('', '', string.punctuation)
    
    for text in texts:
        cleaned_words = []
        for word in text.split():
            # 转换为小写
            word = word.lower()
            # 删除标点符号
            word = word.translate(translator)
            # 删除非英文字符
            word = english_chars_and_whitespace_pattern.sub('', word)
            
            if not word:
                continue
            
            # 提取词干
            # word = stemmer.stem(word)
        
            # 进行词形还原
            word = lemmatizer.lemmatize(word).strip()
            
            # 检查是否是英文单词，且不在停用词列表中
            # if english_word_pattern.search(word) and word not in stopwords:
            #     cleaned_words.append(word)
            
            if word and word not in stopwords:
                cleaned_words.append(word)
            
        # 转换为字符串
        cleaned_text = ' '.join(cleaned_words)
        result.append(cleaned_text)
    
    return result

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def line_processor_use_lemma_plus(text, stopwords):
    # 初始化词形还原器
    lemmatizer = WordNetLemmatizer()
    
    # 定义英文字符和空白字符的正则表达式模式
    english_chars_and_whitespace_pattern = re.compile(r'[^a-zA-Z\s]')
    
    # 创建一个转换表，将所有标点符号映射为None
    translator = str.maketrans('', '', string.punctuation)
    
    cleaned_words = []
    # 先对文本进行分句
    sentences = sent_tokenize(text)
    for sentence in sentences:
        # 分词
        words = word_tokenize(sentence)
        # 词性标注
        pos_tags = pos_tag(words)
        for word, pos in pos_tags:
            # 转换为小写
            word = word.lower()
            # 删除标点符号
            word = word.translate(translator)
            # 删除非英文字符
            word = english_chars_and_whitespace_pattern.sub('', word)
            
            if not word or word in stopwords:
                continue
            
            # 获取单词的词性
            word_pos = get_wordnet_pos(pos)
            # 进行词形还原
            lemmatized_word = lemmatizer.lemmatize(word, pos=word_pos).strip()
            cleaned_words.append(lemmatized_word)
            
    # 转换为字符串
    cleaned_text = ' '.join(cleaned_words)
    return cleaned_text
    

def processor_use_lemma_plus(texts, stopwords):
    result = []
    for text in texts:
        processed_text = line_processor_use_lemma_plus(text, stopwords)
        result.append(processed_text)
    return result