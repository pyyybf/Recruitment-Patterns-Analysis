# 这是一个判断是否为空行的函数，可以直接使用来过滤空行。
import os

def is_not_empty(line):
    return len(line.strip()) > 0

def have_more_than_3_words(line):
    return len(line.split()) > 3

def write_to_file(filepath, data):
    # 检查文件是否存在
    if os.path.exists(filepath):
        # 如果文件存在，删除重写
        os.remove(filepath)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(line + '\n')

def count_txt_files(directory):
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                count += 1
    return count