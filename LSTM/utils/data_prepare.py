import pandas as pd
import numpy as np
import os
from const.paths import topic_paths
import os


def prepare_training_set():
    with open("source/intersection_of_cik_in_training_set.txt") as f:
        cik_in_training_set = [int(line.strip()) for line in f.readlines()]
        cik_in_training_set = set(cik_in_training_set)

    topics = os.listdir(topic_paths)

    save_path = "source/train_set"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for topic in topics:
        print(topic)
        topic_path = os.path.join(topic_paths, topic)
        df = pd.read_csv(topic_path)
        df = df[df['cik'].isin(cik_in_training_set)]
        df = df.sort_values('cik')  # Sort the DataFrame by 'cik' column
        df.to_csv(f"{save_path}/{topic}.csv", index=False)


if __name__ == '__main__':
    prepare_training_set()
