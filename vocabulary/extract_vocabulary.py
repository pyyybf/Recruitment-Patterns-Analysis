import os
from utils.const import paths
from utils.vocabulary import extract_vocabulary
from utils.vocabulary import load_sets_from_file
from utils.vocabulary import save_sets_to_file


def get_vocabulary(read_path, output_path):
    # 遍历read_path下的所有子文件夹（假设它们是年份文件夹）
    for year_folder in os.listdir(read_path):
        year_folder_path = os.path.join(read_path, year_folder)

        # 确保它是一个文件夹
        if os.path.isdir(year_folder_path):

            # 遍历年份文件夹中的所有文件
            for filename in os.listdir(year_folder_path):
                file_path = os.path.join(year_folder_path, filename)

                # 确保它是一个文件
                if os.path.isfile(file_path) and filename.endswith('.txt'):
                    # 读取文件内容
                    with open(file_path, 'r', encoding='utf-8') as file:
                        lines = file.read()

                    # Use extractor to extract vocabulary
                    new_vocabulary_set = extract_vocabulary(lines)

                    vocabulary_path = os.path.join(output_path, f"{os.path.basename(year_folder)}_vocabulary.json")

                    # 确保对应年份的vocabulary_path存在
                    if os.path.exists(vocabulary_path):
                        # load Vocabulary if vocabulary already existed
                        Vocabulary = load_sets_from_file(vocabulary_path)
                    else:
                        Vocabulary = []

                    Vocabulary.append(list(new_vocabulary_set))
                    save_sets_to_file(vocabulary_path, Vocabulary)
                    print(f"Vocabulary Extraction of {file_path} Completed")


if __name__ == '__main__':
    read_path = paths.cleaned_valuable  # 输入文件夹路径
    output_path = paths.vocabulary  # 输出文件夹路径
    get_vocabulary(read_path,output_path)
    print("All Done!")
