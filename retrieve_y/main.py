import csv
import json
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import pre_processor


def retrieve_y():
    # 先读下数据文件
    data_file_name = "recruit_number.csv"
    with open(f"./data/{data_file_name}", "r") as fp:
        lines = [row[0] for row in csv.reader(fp) if len(row) > 0 and len(row[0].strip()) > 0]
    print(len(lines))

    # 预处理
    processed_lines = pre_processor.preprocess_lines(lines)

    # TODO: 生成tfidf vectorizer！！目前先分成两部分试试，应该得有个test？？时间太长的话就混在一起吧。。
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(processed_lines)

    # print(json.dumps(vectorizer.vocabulary_, indent=4))


if __name__ == "__main__":
    retrieve_y()
