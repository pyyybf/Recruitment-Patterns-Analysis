import csv
import os
import pandas as pd
from tqdm import tqdm
from gensim.models import LdaModel
from gensim.corpora import Dictionary


def get_topic_score_by_year(lda_model_path, id2word_path, prefix="train"):
    print("Loading Model...")
    lda_model = LdaModel.load(lda_model_path)
    vocab = Dictionary.load(id2word_path)
    print("Load Model Successfully.\n")

    if not os.path.exists(f"./topic_data/{prefix}_topic"):
        os.makedirs(f"./topic_data/{prefix}_topic")

    topics = lda_model.get_topics()
    headers = ["cik", "year", "file_name"] + [f"Topic {i}" for i in range(0, len(topics))]

    for year in range(2016, 2023):
        file_list = os.listdir(f"./data/{prefix}_data/{year}")
        file_list = sorted(file_list, key=lambda item: int(item.split("_")[0]))
        with open(f"./{prefix}_topic/{year}_topic.csv", "w") as target:
            writer = csv.DictWriter(target, fieldnames=headers)
            writer.writeheader()
            with tqdm(total=len(file_list), unit="file", desc=f"Calculate Topic Scores of {year}") as pbar_topic:
                for file_name in file_list:
                    row_data = {
                        "cik": file_name.split("_")[0],
                        "year": year,
                        "file_name": file_name,
                    }
                    with open(f"./data/{prefix}_data/{year}/{file_name}") as file:
                        text = file.read().split()
                    bow = vocab.doc2bow(text)
                    topic_scores = lda_model.get_document_topics(bow, minimum_probability=0.0)
                    for topic_id, topic_score in topic_scores:
                        row_data[f"Topic {topic_id}"] = topic_score
                    writer.writerow(row_data)
                    pbar_topic.update(1)


def get_topic_score(lda_model_path, id2word_path, prefix="train"):
    print("Loading Model...")
    lda_model = LdaModel.load(lda_model_path)
    vocab = Dictionary.load(id2word_path)
    print("Load Model Successfully.\n")

    topics = lda_model.get_topics()

    with open(f"./data/{prefix}_data/{prefix}_change_rate_unfilled.csv", "r") as fp:
        _len = len(fp.readlines()) - 1

    data = []
    with open(f"./data/{prefix}_data/{prefix}_change_rate_unfilled.csv", "r") as fp:
        rows = csv.DictReader(fp)
        with tqdm(total=_len, unit="file", desc=f"Calculate Topic Scores") as pbar_topic:
            for row in rows:
                row_data = {**row}
                with open(f"./data/{prefix}_data/{row['year']}/{row['file_name']}") as file:
                    text = file.read().split()
                bow = vocab.doc2bow(text)
                topic_scores = lda_model.get_document_topics(bow, minimum_probability=0.0)
                for topic_id, topic_score in topic_scores:
                    row_data[f"Topic {topic_id}"] = topic_score
                data.append(row_data)
                pbar_topic.update(1)

    # new X csv
    headers = ["cik", "year", "file_name", "change_rate"] + [f"Topic {i}" for i in range(0, len(topics))]
    with open(f"./data/{prefix}_data/{prefix}_topics.csv", "w") as fp:
        writer = csv.DictWriter(fp, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)


def prepare_data(file_path="./data/train_data/train_topics.csv", topic_num=20, classifier=False):
    df = pd.read_csv(file_path)

    X = df[[f"Topic {i}" for i in range(0, topic_num)]]
    y = df["change_rate"]
    if classifier:
        y = (y > 0).astype(int)

    return X, y
