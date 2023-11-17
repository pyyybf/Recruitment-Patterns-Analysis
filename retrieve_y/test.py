import os
import random
import shutil
from retrieve_y.utils.retrieval_tool import preprocess_lines
from utils import fs
from main import retrieve_recruit_info, retrieve_file_recruit_info, generate_transformer_doc_inc
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords


def get_samples(n_per_year):
    # 每年抽几个吧
    fs.clear_dir(f"./data_sample")
    for year in [dir for dir in os.listdir("./data_txt") if not dir.startswith(".")]:
        # 删除现有的年份文件夹
        os.mkdir(f"./data_sample/{year}")

        file_names = [file_name for file_name in os.listdir(f"./data_txt/{year}") if not file_name.startswith(".")]
        random.shuffle(file_names)
        for file_name in file_names[:n_per_year]:
            shutil.copy(f"./data_txt/{year}/{file_name}", f"./data_sample/{year}/{file_name}")


if __name__ == "__main__":
    # get_samples(10)
    retrieve_recruit_info("./data_sample")
    # retrieve_file_recruit_info()
    # fs.clear_dir("./retrieve_results_tmp")
    # transformer, doc_inc_mat = generate_transformer_doc_inc()
    # retrieve_file_recruit_info("4969_000000496917000006_d12312016.htm.txt", transformer, doc_inc_mat, top_n=5,
    #                            source_dir="./data_txt/2016",
    #                            target_dir="./retrieve_results_tmp")
    # print(preprocess_lines(["As of both December 31, 2016 and 2015, Credco had 6 employees."],
    #                        stemmer=LancasterStemmer(),
    #                        stop_words=set(stopwords.words("english")),
    #                        required_words={"employee", "employees", "employed", "headcount"}))
