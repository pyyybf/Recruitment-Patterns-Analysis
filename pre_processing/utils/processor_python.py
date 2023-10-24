import os
import re
from bs4 import BeautifulSoup
from utils.helpers import is_not_empty
from utils.helpers import have_more_than_3_words

def preprocess_line(line, current_paragraph):
    # 去除前后空白
    processed_line = line.strip()
    # 去除html标签，将html实体转换为对应的字符
    processed_line = BeautifulSoup(processed_line, "html.parser").get_text()
    
    # 去除非字母数字字符
    processed_line = re.sub(r'[^a-zA-Z0-9\s]', '', processed_line)
    # 去除非单词的字母组合
    processed_line = re.sub(r'\b[a-zA-Z]{1,2}\b', '', processed_line)
    
    processed_line = re.sub(r'\s+', ' ', processed_line)
    processed_line = processed_line.strip()
    
    if processed_line:
        if processed_line[0].islower():
            current_paragraph += ' ' + processed_line
        else:
            if current_paragraph:
                return processed_line, current_paragraph
            current_paragraph = processed_line
    return current_paragraph, None

def process_file(filepath):
    paragraphs = []
    current_paragraph = ''
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            current_paragraph, paragraph = preprocess_line(line, current_paragraph)
            if paragraph is not None and have_more_than_3_words(paragraph):
                paragraphs.append(paragraph)
    
    # 处理文件最后的段落
    if is_not_empty(current_paragraph)and have_more_than_3_words(current_paragraph):
        paragraphs.append(current_paragraph)
        
    return paragraphs

def process_directory(directory_path):
    all_paragraphs = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.txt'):
                filepath = os.path.join(root, file)
                paragraphs = process_file(filepath)
                all_paragraphs.extend(paragraphs)
    return all_paragraphs
