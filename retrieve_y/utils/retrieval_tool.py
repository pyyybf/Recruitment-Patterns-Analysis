import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer


def split_paragraph(lines):
    split_lines = []
    # 姑且试试按英语句号split一下吧
    for line in lines:
        sentences = re.sub(r"&.{0,5};", " ", line)
        sentences = sentences.split(". ")
        for sentence in sentences:
            if len(sentence.strip()) > 0:
                split_lines.append(sentence.strip())
    return split_lines


def merge_paragraph(lines):
    processed_lines = []
    cur_paragraph = ""
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue
        if line[0].isupper() and len(cur_paragraph) > 0:
            processed_lines.append(cur_paragraph.strip())
            cur_paragraph = ""
        cur_paragraph += line
        cur_paragraph += " "
    processed_lines.append(cur_paragraph.strip())
    return processed_lines


def retrieve_top_n_idx(doc_inc_mat, query_inc_mat, top_n=5):
    cs = np.dot(query_inc_mat, doc_inc_mat.T)
    max_sims = np.max(cs, axis=1)

    # Get index list of top-n positive value
    positive_idx_list = np.where(max_sims > 0)[0]
    sorted_positive_idx_list = np.argsort(max_sims[positive_idx_list])
    doc_idx_list = positive_idx_list[sorted_positive_idx_list][:-top_n - 1:-1]

    return doc_idx_list


def preprocess_lines(lines, stemmer=None, stop_words=None, required_words=None):
    processed_lines = []
    for line in lines:
        processed_line = line.lower().strip()

        # 先去除所有除非字母字符
        processed_line = re.sub(r"&.{0,5};", " ", processed_line)
        processed_line = re.sub(r"[^a-zA-Z-\s]+", " ", processed_line)
        processed_line = re.sub(r"\s+", " ", processed_line)
        processed_line = processed_line.strip()

        # 分词
        processed_line = word_tokenize(processed_line)

        # 筛选必须词
        if required_words:
            for required_word in required_words:
                if required_word in processed_line:
                    break
            else:
                processed_line = []

        # 去除停用词+取一下词根
        if stemmer and stop_words:
            processed_line = [stemmer.stem(word) for word in processed_line if word not in stop_words]
        elif stemmer:
            processed_line = [stemmer.stem(word) for word in processed_line]
        elif stop_words:
            processed_line = [word for word in processed_line if word not in stop_words]

        processed_lines.append(processed_line)
    return processed_lines


def parse_vocabulary(processed_lines):
    vocab = set()
    for line in processed_lines:
        for word in line:
            vocab.add(word)
    vocab = sorted(vocab)
    return vocab


def lines2matrix(processed_lines, vocabulary, keywords, min_len=5):
    incidence_mat = []
    for line in processed_lines:
        if len(line) < min_len:
            incidence_mat.append([0] * len(vocabulary))
            continue
        inc = 1 / (len(line) + 1)
        for keyword in keywords:
            if keyword in line:
                incidence_mat.append([inc] * len(vocabulary))
                break
        else:
            incidence_mat.append([inc if word in line else 0 for word in vocabulary])
    incidence_mat = np.array(incidence_mat)
    return incidence_mat


class Lines2Matrix:
    """Convert a collection of raw lines to a matrix of incidence features.

    Parameters
    ----------
    stop_words : {'english', 'german', 'indonesia', 'portuguese', 'spanish'}, list, default=None
    stemmer : {'Lancaster', 'Porter', 'Snowball'}, default=None
    required_words : list, default=None
    """

    def __init__(self, *,
                 stop_words=None,
                 stemmer=None,
                 required_words=None,
                 keywords=None,
                 min_len=5):
        self.vocabulary = set()
        if type(stop_words) == list:
            self.stop_words = set(stop_words)
        elif type(stop_words) == str:
            self.stop_words = set(stopwords.words(stop_words))
        else:
            self.stop_words = set()
        if stemmer == "Lancaster":
            self.stemmer = LancasterStemmer()
        elif stemmer == "Porter":
            self.stemmer = PorterStemmer()
        elif stemmer == "Snowball":
            self.stemmer = SnowballStemmer("english")
        else:
            self.stemmer = None
        self.required_words = set(required_words or [])
        self.keywords = set(keywords or [])
        self.min_len = min_len

    def fit(self, lines):
        processed_lines = preprocess_lines(lines)
        self.vocabulary = parse_vocabulary(processed_lines)
        return self

    def fit_transform(self, lines):
        # Pre-process the input lines
        processed_lines = preprocess_lines(lines, self.stemmer, self.stop_words)
        # Get vocabulary from processed lines
        self.vocabulary = parse_vocabulary(processed_lines)
        # Generate the incidence matrix
        incidence_mat = lines2matrix(processed_lines, self.vocabulary, self.keywords, self.min_len)
        return incidence_mat

    def transform(self, lines):
        processed_lines = preprocess_lines(lines, self.stemmer, self.stop_words, self.required_words)
        incidence_mat = lines2matrix(processed_lines, self.vocabulary, self.keywords, self.min_len)
        return incidence_mat
