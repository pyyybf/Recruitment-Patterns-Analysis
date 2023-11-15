import csv
import os
from tqdm import tqdm

from utils import fs
from utils.retrieval_tool import Lines2Matrix, retrieve_top_n_idx, split_paragraph, merge_paragraph


def retrieve_y(base_dir="data_txt", top_n=5):
    # TODO: 目前先分成两部分试试 应该得有个test？？时间太长的话就混在一起吧。。
    # 先读下数据文件
    data_file_name = "recruit_number.csv"
    with open(f"./recruit_text/{data_file_name}", "r") as fp:
        lines = [row[0] for row in csv.reader(fp) if len(row) > 0 and len(row[0].strip()) > 0]

    # 读取必须词
    with open("required_words.txt", "r") as fp:
        required_words = fp.read().strip().split()

    # TODO: 要用word2vec吗 感觉没啥必要 先试试效果吧 不行再换好了
    transformer = Lines2Matrix(stop_words="english",
                               stemmer="Lancaster",
                               required_words=required_words,
                               keywords={"full-time", "part-time"})
    doc_inc_mat = transformer.fit_transform(lines)

    # TODO: 好 开始遍历文件夹了
    fs.clear_dir("./retrieve_results")
    for year in range(2016, 2023):
        os.mkdir(f"./retrieve_results/{year}")

        file_names = fs.list_dir(f"./{base_dir}/{year}")
        with tqdm(total=len(file_names), unit="file",
                  desc=f"Retrieve Recruitment Information from Files of {year}") as pbar_retrieve:
            for file_name in file_names:
                with open(f"./{base_dir}/{year}/{file_name}", "r") as fp:
                    lines = [line.strip() for line in fp.readlines() if len(line.strip()) > 0]

                # 先把莫名分段拼回去
                lines = merge_paragraph(lines)
                # 再分个句 不行就去掉 泪目
                lines = split_paragraph(lines)

                cur_inc_mat = transformer.transform(lines)
                idxs = retrieve_top_n_idx(doc_inc_mat, cur_inc_mat, top_n=top_n)

                with open(f"./retrieve_results/{year}/retrieved_{file_name}", "w") as fp:
                    for idx in idxs:
                        fp.write(f"{lines[idx]}\n")
                pbar_retrieve.update(1)


if __name__ == "__main__":
    retrieve_y(top_n=10)
