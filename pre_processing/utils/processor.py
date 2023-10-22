import dask.bag as db
import os
import re
from bs4 import BeautifulSoup

# 这是一个判断是否为空行的函数，可以直接使用来过滤空行。
def is_not_empty(line):
    return len(line.strip()) > 0

# 这是一个预处理函数，对每一行进行预处理，可以根据实际情况进行修改
def preprocess_line(line):
    # 添加实际的预处理代码, 因为数据量较大, 所以使用dask.bag将数据一行一行进行处理，最后返回一个bag
    processed_line = line.strip()  # 去除前后空白
    # 还可以添加其他的预处理操作
    
    # 去除html标签，将html实体转换为对应的字符
    processed_line = BeautifulSoup(processed_line, "html.parser").get_text()
    # 去除非字母数字字符
    processed_line = re.sub(r'[^a-zA-Z0-9\s]', '', processed_line)
    # 去除非单词的字母组合
    processed_line = re.sub(r'\b[a-zA-Z]{1,2}\b', '', processed_line)
    
    return processed_line

# 这是一个处理单个文件的函数，可以直接使用来处理单个文件
def process_file(filepath):
    b = db.read_text(filepath)
    processed_bag = b.map(preprocess_line).filter(is_not_empty) # 过滤空行，减少数据量
    return processed_bag

# 这是一个处理整个目录的函数，可以直接使用来处理整个目录
def process_directory(directory_path):
    all_processed_data = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.txt'):
                filepath = os.path.join(root, file)
                processed_bag = process_file(filepath)
                all_processed_data.append(processed_bag)
    # 合并所有处理后的数据
    return db.concat(all_processed_data)
