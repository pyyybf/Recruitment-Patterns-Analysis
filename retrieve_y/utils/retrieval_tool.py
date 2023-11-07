import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer


def retrieve_top_n_idx(doc_inc_mat, query_inc_mat, top_n=5):
    cs = np.dot(query_inc_mat, doc_inc_mat.T)
    max_sims = np.max(cs, axis=1)

    # Get index list of top-n positive value
    positive_idx_list = np.where(max_sims > 0)[0]
    sorted_positive_idx_list = np.argsort(max_sims[positive_idx_list])[::-1]
    doc_idx_list = positive_idx_list[sorted_positive_idx_list][:top_n]

    return doc_idx_list


class Lines2Matrix:
    """Convert a collection of raw lines to a matrix of incidence features.

    Parameters
    ----------
    stop_words : {'english'}, list, default=None
    stemmer : {'Lancaster', 'Porter', 'Snowball'}, default='Lancaster'
    """

    def __init__(self, *, stop_words=None, stemmer="Lancaster"):
        self.vocabulary = set()
        if type(stop_words) == list:
            self.stop_words = set(stop_words)
        elif type(stop_words) == str:
            self.stop_words = set(stopwords.words("english"))
        else:
            self.stop_words = set()
        if stemmer == "Porter":
            self.stemmer = PorterStemmer()
        elif stemmer == "Snowball":
            self.stemmer = SnowballStemmer("english")
        else:
            self.stemmer = LancasterStemmer()

    def preprocess_lines(self, lines):
        processed_lines = []
        for line in lines:
            processed_line = line.strip()

            # 先去除所有除非字母字符
            processed_line = re.sub(r"[^a-zA-Z\s]+", " ", processed_line)
            processed_line = re.sub(r"\s+", " ", processed_line)
            processed_line = processed_line.strip()

            # 取一下词根
            processed_line = word_tokenize(processed_line)
            processed_line = [self.stemmer.stem(word) for word in processed_line if word not in self.stop_words]

            # TODO: 限制下词频 感觉财年之类的词频率太高了

            processed_lines.append(processed_line)
        return processed_lines

    def fit_transform(self, lines):
        # Pre-process the input lines
        processed_lines = self.preprocess_lines(lines)

        # Get vocabulary from processed lines
        vocab = set()
        for line in processed_lines:
            for word in line:
                vocab.add(word)
        self.vocabulary = sorted(vocab)

        # Generate the incidence matrix
        incidence_mat = []
        for line in processed_lines:
            incidence_mat.append([1 if word in line else 0 for word in self.vocabulary])

        return np.array(incidence_mat)

    def fit(self, lines):
        processed_lines = self.preprocess_lines(lines)
        vocab = set()
        for line in processed_lines:
            for word in line:
                vocab.add(word)
        self.vocabulary = sorted(vocab)
        return self

    def transform(self, lines):
        processed_lines = self.preprocess_lines(lines)
        incidence_mat = []
        for line in processed_lines:
            incidence_mat.append([1 if word in line else 0 for word in self.vocabulary])
        return np.array(incidence_mat)
