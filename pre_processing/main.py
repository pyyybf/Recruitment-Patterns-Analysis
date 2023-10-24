from utils import processor_python
from utils import processor_dask
from const import paths
from utils.helpers import write_to_file
import os
from tqdm import tqdm
# from utils.helpers import count_txt_files

def main():
    root_path = paths.data_path
    target_root = paths.target_path
    os.makedirs(target_root, exist_ok=True)
    
    dirs = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    with tqdm(total=len(dirs), unit='dir', desc="Processing Directories") as pbar_dirs:
        for dir in dirs:
            print(f"Processing {dir}...")

            dir_path = os.path.join(root_path, dir)
            target_year_root = os.path.join(target_root, dir)
            os.makedirs(target_year_root, exist_ok=True)

            files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f.endswith('.txt')]
            with tqdm(total=len(files), unit='file', desc=f"Processing {dir}", leave=False) as pbar_files:
                for file in files:
                    filepath = os.path.join(dir_path, file)
                    paragraphs = processor_python.process_file(filepath)
                    target_filepath = os.path.join(target_year_root, 'processed_' + file)
                    write_to_file(target_filepath, paragraphs)
                    pbar_files.update(1)
            pbar_dirs.update(1)



if __name__ == '__main__':
    main()
    

# def main():
#     root_path = paths.data_path
#     target_root = paths.target_path
#     os.makedirs(target_root, exist_ok=True)
    
#     for dir in os.listdir(root_path):
#         dir_path = os.path.join(root_path, dir)
        
#         if not os.path.isdir(dir_path):
#             continue
        
#         print(f"Processing {dir} 10k files...")
        
#         target_year_root = os.path.join(target_root, dir)
#         os.makedirs(target_year_root, exist_ok=True)
        
#         total_files = count_txt_files(dir_path)
#         with tqdm(total=total_files, unit='file', desc=f"Processing {dir}") as pbar:
#             for file in os.listdir(dir_path):
#                 filepath = os.path.join(dir_path, file)
#                 if os.path.isfile(filepath) and file.endswith('.txt'):
#                     try:
#                         paragraphs = processor_dask.process_file(filepath)
#                         target_filepath = os.path.join(target_year_root, 'processed_' + file)
#                         write_to_file(target_filepath, paragraphs)
#                     except Exception as e:
#                         print(f"Error processing file {file}: {e}")
#                     pbar.update(1)
    # for root, dirs, files in os.walk(root_path):
    #     for file in files:
    #         if file.endswith('.txt'):
    #             filepath = os.path.join(root, file)
    #             paragraphs = processor_python.process_file(filepath)
    #             target_filepath = os.path.join(target_root, 'processed_' + file)
    #             write_to_file(target_filepath, paragraphs)
    #         pbar.update(1)
    # pbar.close()

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