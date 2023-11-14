import os
import numpy as np
import pandas as pd
from utils.const import paths
from utils.pre_processor import processor_use_lemma_plus as processor


def clean_txt(read_path, output_path, Stopwords=None) -> None:
    # 遍历read_path下的所有子文件夹（假设它们是年份文件夹）
    for year_folder in os.listdir(read_path):
        year_folder_path = os.path.join(read_path, year_folder)

        # 确保它是一个文件夹
        if os.path.isdir(year_folder_path):
            # 创建相应的输出文件夹，如果它不存在
            output_year_folder = os.path.join(output_path, year_folder)
            if not os.path.exists(output_year_folder):
                os.makedirs(output_year_folder)

            # 遍历年份文件夹中的所有文件
            for filename in os.listdir(year_folder_path):
                file_path = os.path.join(year_folder_path, filename)

                # 确保它是一个文件
                if os.path.isfile(file_path) and filename.endswith('.txt'):
                    # 读取文件内容
                    with open(file_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()

                    # 使用processor处理文本
                    cleaned_content = processor(lines, Stopwords)

                    # 保存处理后的内容到输出文件夹
                    output_file_path = os.path.join(
                        output_year_folder, filename)
                    with open(output_file_path, 'w', encoding='utf-8') as output_file:
                        output_file.write(cleaned_content)


if __name__ == '__main__':
    read_path = paths.valuable_data_original  # 输入文件夹路径
    output_path = paths.valuable_data_cleaned  # 输出文件夹路径
    clean_txt(read_path, output_path)
