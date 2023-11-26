import csv
import os
from tqdm import tqdm

from utils import fs, paths
from utils.match_tool import match_employee_num
from utils.retrieval_tool import Lines2Matrix, retrieve_top_n_idx, split_paragraph, merge_paragraph
from utils.get_employee import get_employee, get_change_rate


def retrieve_recruit_info(base_dir=paths.data_txt_dir, top_n=5):
    # Read labeled recruitment information sentences
    data_file_name = "recruit_number.csv"
    with open(f"./recruit_text/{data_file_name}", "r") as fp:
        lines = [row[0] for row in csv.reader(fp) if len(row) > 0 and len(row[0].strip()) > 0]

    # Read required words, lines that do not contain any of the required words will be eliminated
    with open("./my_words/required_words.txt", "r") as fp:
        required_words = fp.read().strip().split()
    # Read keywords, lines that contain any of the keywords will be retrieved
    with open("./my_words/keywords.txt", "r") as fp:
        keywords = fp.read().strip().split()

    transformer = Lines2Matrix(stop_words="english",
                               stemmer="Lancaster",
                               required_words=required_words,
                               keywords=keywords,
                               min_len=3)
    doc_inc_mat = transformer.fit_transform(lines)

    fs.clear_dir("./retrieve_results")
    for year in range(2016, 2023):
        os.mkdir(f"./retrieve_results/{year}")

        file_names = fs.list_dir(f"{base_dir}/{year}")
        with tqdm(total=len(file_names), unit="file",
                  desc=f"Retrieve Recruitment Information from Files of {year}") as pbar_retrieve:
            for file_name in file_names:
                with open(f"{base_dir}/{year}/{file_name}", "r") as fp:
                    lines = [line.strip() for line in fp.readlines() if len(line.strip()) > 0]

                # Remove incorrect subparagraph
                lines = merge_paragraph(lines)
                # Split paragraph into sentences
                lines = split_paragraph(lines)

                cur_inc_mat = transformer.transform(lines)
                idxs = retrieve_top_n_idx(doc_inc_mat, cur_inc_mat, top_n=top_n)

                with open(f"./retrieve_results/{year}/retrieved_{file_name}", "w") as fp:
                    for idx in idxs:
                        fp.write(f"{lines[idx]}\n")
                pbar_retrieve.update(1)


def match_recruit_info(base_dir, target_file, record_file):
    # Read ban words, lines that contain any ban word will be eliminated
    with open("./my_words/ban_words.txt", "r") as fp:
        ban_words = set([line.strip() for line in fp.readlines()])

    # Clear the record files
    with open(record_file, "w") as fp:
        fp.write("")
    with open(target_file, "w") as fp:
        fp.write("year,file_name,employee_num\n")

    for year in range(2016, 2023):
        file_names = fs.list_dir(f"{base_dir}/{year}")
        with tqdm(total=len(file_names), unit="file",
                  desc=f"{year}") as pbar_retrieve:
            for file_name in file_names:
                with open(f"{base_dir}/{year}/{file_name}", "r") as fp:
                    lines = [line.strip() for line in fp.readlines() if len(line.strip()) > 0]

                # Remove incorrect subparagraph
                lines = merge_paragraph(lines)
                # Split paragraph into sentences
                lines = split_paragraph(lines)

                employee_num = match_employee_num(lines, year, record_file, ban_words)

                if employee_num >= 0:
                    with open(target_file, "a") as fp:
                        fp.write(f"{year},{file_name},{employee_num}\n")

                pbar_retrieve.update(1)


if __name__ == "__main__":
    # Clean output files
    fs.clear_dir(paths.output_dir)
    # Retrieve recruitment info with tf-idf matrix, failed
    # retrieve_recruit_info(top_n=10)
    # Match recruitment info with regular expression into number_match.csv, record matched lines in employee_lines.txt
    match_recruit_info(base_dir=paths.data_txt_dir,
                       target_file=f"{paths.output_dir}/number_match.csv",
                       record_file=f"{paths.output_dir}/employee_lines.txt")
    # Generate employee_num.json from number_match.csv to easily get number by cik and year
    get_employee(csv_file=f"{paths.output_dir}/number_match.csv", output_file=f"{paths.output_dir}/employee_num.json")
    # Calculate change rates and generate change_rate.csv
    get_change_rate(source_file=f"{paths.output_dir}/employee_num.json",
                    target_file=f"{paths.output_dir}/change_rate.csv",
                    tmp_output_dir=paths.output_dir)
