import csv
import os
import re
from num2words import num2words
from word2number import w2n
from tqdm import tqdm

from retrieve_y.get_employee import get_employee
from utils import fs
from utils.retrieval_tool import Lines2Matrix, retrieve_top_n_idx, split_paragraph, merge_paragraph


def retrieve_recruit_info(base_dir="data_txt", top_n=5):
    # TODO: 目前先分成两部分试试
    # 先读下数据文件
    data_file_name = "recruit_number.csv"
    with open(f"./recruit_text/{data_file_name}", "r") as fp:
        lines = [row[0] for row in csv.reader(fp) if len(row) > 0 and len(row[0].strip()) > 0]

    # 读取必须词 必须包含列表词之一否则0分
    with open("required_words.txt", "r") as fp:
        required_words = fp.read().strip().split()
    # 读取关键词 包含就分很高
    with open("keywords.txt", "r") as fp:
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


def convert_number(num_str):
    if num_str.isdigit():
        return int(num_str)
    elif num_str == "no":
        return 0
    elif "(" in num_str:
        numbers = num_str.split("(")
        num1 = numbers[0].strip()
        num2 = numbers[1].strip(")").strip()
        if num1.isdigit():
            return int(num1)
        elif num2.isdigit():
            return int(num2)
        num1 = convert_number(num1)
        num2 = convert_number(num2)
        if num1:
            return num1
        else:
            return num2
    try:
        return w2n.word_to_num(num_str)
    except Exception as e:
        return -1


def match_recruit_info(base_dir="data_txt", target_file="./number_match.csv"):
    numbers = r"[\d, ()]+"
    english_numbers = [num2words(i) for i in range(21)]
    verbs = ["have", "has", "had", "employed", "employ", "employs", "consisted of", "consists of", "made up of",
             "makes of"]
    nouns = ["employee", "employees", "headcount", "teammates", "people", "persons"]

    pattern = fr"({'|'.join(verbs)})\s+([a-z]+\s)*?({numbers}|no|{'|'.join(english_numbers)})\s+([a-z-,]+\s)*?(?:{'|'.join(nouns)})"

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

                employee_nums = []
                for line in lines:
                    if "employ" not in line:
                        continue
                    res = re.findall(pattern, line)
                    if len(res) > 0:
                        employee_num = res[0][2].strip().replace(",", "")
                        employee_num = re.sub(r"\s+", "", employee_num)
                        employee_nums.append(convert_number(employee_num))

                if len(employee_nums) > 0:
                    with open(target_file, "a") as fp:
                        fp.write(f"{year},{file_name},{max(employee_nums)}\n")

                pbar_retrieve.update(1)


if __name__ == "__main__":
    # retrieve_recruit_info(top_n=10)
    match_recruit_info()
    get_employee(csv_file="./number_match.csv", output_file="./employee_num.json")
