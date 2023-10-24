from utils import processor_python
from utils import processor_dask
from const import paths
from utils.helpers import write_to_file
import os

def main():
    root_path = paths.year_path
    target_root = paths.target_path
    os.makedirs(target_root, exist_ok=True)
    
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('.txt'):
                filepath = os.path.join(root, file)
                paragraphs = processor_python.process_file(filepath)
                target_filepath = os.path.join(target_root, 'processed_' + file)
                write_to_file(target_filepath, paragraphs)

    #processed_list = processor_python.process_file(test_filepath)
    

# def main():
#     # 文件路径
#     filepath = "/Users/weichentao/Documents/USC/2023fall/540/project/data_txt/2016"
    
#     processed_bag = processor_dask.process_file(filepath)
    
#     # 计算处理后的总行数
#     res = processed_bag.compute()
#     for i, line in enumerate(res):
#         if i >= 1000:
#             break
#         print(line)

if __name__ == '__main__':
    main()
    main()