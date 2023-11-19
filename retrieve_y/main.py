import csv
import os
from tqdm import tqdm

from utils import fs
from utils.match_tool import match_employee_num
from utils.retrieval_tool import Lines2Matrix, retrieve_top_n_idx, split_paragraph, merge_paragraph
from utils.get_employee import get_employee


def retrieve_recruit_info(base_dir="data_txt", top_n=5):
    # TODO: 目前先分成两部分试试
    # 先读下数据文件
    data_file_name = "recruit_number.csv"
    with open(f"./recruit_text/{data_file_name}", "r") as fp:
        lines = [row[0] for row in csv.reader(fp) if len(row) > 0 and len(row[0].strip()) > 0]

    # 读取必须词 必须包含列表词之一否则0分
    with open("./my_words/required_words.txt", "r") as fp:
        required_words = fp.read().strip().split()
    # 读取关键词 包含就分很高
    with open("./my_words/keywords.txt", "r") as fp:
        keywords = fp.read().strip().split()

    transformer = Lines2Matrix(stop_words="english",
                               stemmer="Lancaster",
                               required_words=required_words,
                               keywords=keywords,
                               min_len=3)
    doc_inc_mat = transformer.fit_transform(lines)

    # 开始遍历文件夹
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


def match_recruit_info(base_dir="data_txt", target_file="./number_match.csv", record_file="./employee_lines.txt"):
    with open("./my_words/ban_words.txt", "r") as fp:
        ban_words = set([line.strip() for line in fp.readlines()])

    # 清理下记录文件
    with open(record_file, "w") as fp:
        fp.write("")
    with open(target_file, "w") as fp:
        fp.write("year,file_name,employee_num\n")

    for year in range(2016, 2023):
        file_names = fs.list_dir(f"./{base_dir}/{year}")
        with tqdm(total=len(file_names), unit="file",
                  desc=f"{year}") as pbar_retrieve:
            for file_name in file_names:
                with open(f"./{base_dir}/{year}/{file_name}", "r") as fp:
                    lines = [line.strip() for line in fp.readlines() if len(line.strip()) > 0]

                # 先把莫名分段拼回去
                lines = merge_paragraph(lines)
                # 再分个句 不行就去掉 泪目
                lines = split_paragraph(lines)

                employee_num = match_employee_num(lines, year, record_file, ban_words)

                if employee_num >= 0:
                    with open(target_file, "a") as fp:
                        fp.write(f"{year},{file_name},{employee_num}\n")

                pbar_retrieve.update(1)


if __name__ == "__main__":
    # retrieve_recruit_info(top_n=10)
    match_recruit_info()
    get_employee(csv_file="./number_match.csv", output_file="./employee_num.json")
