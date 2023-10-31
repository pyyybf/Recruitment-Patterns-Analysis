from utils import vectorizer_dask as vd
from utils import vectorizer_pandas as vp
import pandas as pd
import numpy as np
from utils.const import paths
import os
from utils import merge_csv
from utils.const.stopwords import STOPWORDS
from utils.pre_processor import processor_use_lemma_plus as processor

def clean_data():
    # 读取数据路径和保存路径
    read_path = paths.labeled_data
    save_path = paths.saved_data_cleaned
    
    df = merge_csv.merge_csv(read_path)
    df['cleaned_text'] = processor(df['text'].tolist(), STOPWORDS)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if os.path.exists(os.path.join(save_path, 'cleaned_merged_training_data.csv')):
        os.remove(os.path.join(save_path, 'cleaned_merged_training_data.csv'))
        
    df.to_csv(os.path.join(save_path, 'cleaned_merged_training_data.csv'), index=False)


def vectorize_data():
    file_path = os.path.join(paths.saved_data_cleaned, 'cleaned_merged_training_data.csv')
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    
        labels = df['sentiment']
    
        incidence_matrix = vp.incidence_matrix_generator(df, 'cleaned_text')
    
        tf_idf = vp.tf_idf_generator(df, 'cleaned_text')
    
        incidence_matrix = pd.concat([incidence_matrix, labels], axis=1)
    
        tf_idf = pd.concat([tf_idf, labels], axis=1)
    
        tf_idf.to_csv(os.path.join(paths.saved_data_cleaned, 'tf_idf.csv'), index=False)
    
        incidence_matrix.to_csv(os.path.join(paths.saved_data_cleaned, 'incidence_matrix.csv'), index=False)
    else:
        print("还没有清理好的数据")

if __name__ == '__main__':
    clean_data()
    vectorize_data()
    