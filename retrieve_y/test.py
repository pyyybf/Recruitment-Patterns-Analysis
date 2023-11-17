import os
import random
import shutil
from utils import fs
from main import retrieve_recruit_info


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
