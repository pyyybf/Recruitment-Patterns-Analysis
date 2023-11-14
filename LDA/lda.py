import os
from utils.pre_processor import processor_use_lemma_plus as processor
from utils.const.stopwords import STOPWORDS
from gensim.models import LdaModel
from gensim import corpora
import gensim
from utils.const import paths

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import sys
from typing import List


def split_documents(documents: List[str]) -> List[List[str]]:
    # 分割每个文档中的单词，并将结果存储在列表中
    return [document.split() for document in documents]


def read_documents_from_folder(folder_path: str) -> List[str]:
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                lines = file.readlines()
                cleaned_text = processor(lines, STOPWORDS)
                documents.extend(cleaned_text)
    return documents


def find_lda(texts: List[str], n_topics=10) -> None:
    texts = split_documents(texts)
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    ldamodel = LdaModel(corpus, num_topics=n_topics,
                        id2word=dictionary, passes=20)

    topics = ldamodel.print_topics(num_words=10)
    for topic in topics:
        print(topic)


if __name__ == '__main__':
    folder_path = paths.valuable_data_2021
    documents = read_documents_from_folder(folder_path)
    find_lda(documents)
