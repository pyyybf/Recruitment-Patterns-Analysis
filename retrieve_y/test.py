import os
import random
import shutil

from utils.get_employee import get_employee, get_change_rate
from utils import fs
from main import retrieve_recruit_info, match_recruit_info


def get_samples(n_per_year, sample_dir="./data_sample"):
    # 每年抽几个吧
    fs.clear_dir(sample_dir)
    for year in [dir for dir in os.listdir("./data_txt") if not dir.startswith(".")]:
        # 删除现有的年份文件夹
        os.mkdir(f"./data_sample/{year}")

        file_names = [file_name for file_name in os.listdir(f"./data_txt/{year}") if not file_name.startswith(".")]
        random.shuffle(file_names)
        for file_name in file_names[:n_per_year]:
            shutil.copy(f"./data_txt/{year}/{file_name}", f"{sample_dir}/{year}/{file_name}")


if __name__ == "__main__":
    get_samples(10)
    # retrieve_recruit_info("./data_sample")
    match_recruit_info(base_dir="./data_sample",
                       target_file="./test_number_match.csv",
                       record_file="./test_employee_lines.txt")
    get_employee(csv_file="./test_number_match.csv", output_file="./test_employee_num.json")
    get_change_rate(source_file="./test_employee_num.json", target_file="./test_change_rate.csv")
