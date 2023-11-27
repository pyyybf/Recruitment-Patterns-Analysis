import pandas as pd
import numpy as np
import joblib
from utils.const import paths
import os
import glob
import tqdm
from utils.helper import count_files_in_directory
from utils.pre_processor import processor_use_lemma_plus as processor
from utils.const.stopwords import STOPWORDS

def data_cleaning(total_folder, year_folder, output_folder_path,vectorizer, model):
    
    file_num = count_files_in_directory(os.path.join(total_folder, year_folder))
    dealed_file_num = 0
    
    for txt_file in glob.glob(os.path.join(input_folder_path, year_folder,'*.txt')):
        dealed_file_num += 1
        print(f"{year_folder} process finished: {dealed_file_num}/{file_num}")
        
        with open(txt_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
        clean_lines = processor(lines, STOPWORDS)
            
        X = vectorizer.transform(clean_lines)
        
        predictions = model.predict(X)
        
        retained_lines = [line for line, prediction in zip(lines, predictions) if prediction == 1 or prediction == 'Yes']
        
        if retained_lines:
            # 构建新的文件路径
            base_name = os.path.basename(txt_file)
            output_file_path = os.path.join(output_folder_path, year_folder ,base_name)
            
            os.makedirs(os.path.join(output_folder_path,year_folder), exist_ok=True)
            
            if os.path.isfile(output_file_path):
                print("File already exists, deleting...")
                os.remove(output_file_path)

            # 写入保留的行到新文件
            with open(output_file_path, 'w', encoding='utf-8') as file:
                file.writelines(retained_lines)
            
        
        

if __name__ == '__main__':
    model_path = os.path.join(paths.saved_models, f"DecisionTree.joblib")
    vectorizer_path_tfidf = os.path.join(paths.saved_models, 'tfidf_vectorizer.joblib')
    
    input_folder_path = paths.original_data
    output_folder_path = paths.valuable_data
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path_tfidf)
    
    for year_folder in os.listdir(input_folder_path):
        full_path = os.path.join(input_folder_path, year_folder)
        if os.path.isdir(full_path):
            data_cleaning(total_folder=input_folder_path, year_folder=year_folder, output_folder_path=output_folder_path, vectorizer=vectorizer, model=model)
    # for foldername, subfolders, filenames in os.walk(input_folder_path):
    #     print("Currently dealing with folder: " + foldername)
    #     data_cleaning(total_folder=input_folder_path, year_folder=foldername, output_folder_path= output_folder_path,vectorizer=vectorizer, model = model)
    
    print("Done!")
