from tqdm import tqdm
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer

stop_words = set(stopwords.words("english"))
lanca_stemmer = LancasterStemmer()


def preprocess_line(line):
    processed_line = line.strip()

    # 先去除所有除非字母字符
    processed_line = re.sub(r"[^a-zA-Z\s]+", " ", processed_line)
    processed_line = re.sub(r"\s+", " ", processed_line)
    processed_line = processed_line.strip()

    # 取一下词根
    processed_line = word_tokenize(processed_line)
    processed_line = [lanca_stemmer.stem(word) for word in processed_line if word not in stop_words]
    processed_line = " ".join(processed_line)

    return processed_line


def preprocess_lines(lines):
    processed_lines = []
    with tqdm(total=len(lines), unit="line", desc="Pre-process Lines") as pbar_preprocess:
        for line in lines:
            processed_lines.append(preprocess_line(line))
            pbar_preprocess.update(1)
    return processed_lines
