import os
from utils.pre_processor import processor_use_lemma_plus as processor
from utils.const.stopwords import STOPWORDS
from gensim.models import LdaModel
from gensim import corpora
import gensim
from utils.const import paths
from nltk.tokenize import word_tokenize
from utils.const.stopwords import html_stop_words

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import sys
from typing import List


def split_documents(documents: List[str]) -> List[List[str]]:
    # 分割每个文档中的单词，并将结果存储在列表中
    return [document.split() for document in documents]


def read_documents_from_folder(folder_path: str, stop_words=None) -> List[str]:
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                lines = file.readlines()
                # cleaned_text = processor(lines, STOPWORDS)
                cleaned_texts = []
                for line in lines:
                    words = word_tokenize(line)
                    cleaned_line = ' '.join(
                        [word for word in words if word.lower() not in stop_words])
                    cleaned_texts.append(cleaned_line)
                documents.extend(cleaned_texts)
    return documents


def find_lda(texts: List[str], n_topics: int = 20, save: bool = True, save_path: str = None) -> None:
    texts = split_documents(texts)
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    ldamodel = LdaModel(corpus, num_topics=n_topics,
                        id2word=dictionary, passes=10)
    if save:
        ldamodel.save(save_path)
    topics = ldamodel.print_topics(num_words=10)
    for topic in topics:
        print(topic)


# This is the code for lda using all data.
if __name__ == '__main__':
    total_folder_path = paths.all_data
    year_folder = os.listdir(total_folder_path)
    total_documents = []
    for year in year_folder:
        folder_path = os.path.join(total_folder_path, year)
        documents = read_documents_from_folder(folder_path, html_stop_words)
        total_documents.extend(documents)
    find_lda(total_documents, 20, save=True,
             save_path=paths.lda_model_save_path)

# This is the code for lda using single year data.
# if __name__ == '__main__':
#     folder_path = paths.valuable_data_2017
#     documents = read_documents_from_folder(folder_path)
#     find_lda(documents)
