from utils import vectorizer_dask as vd
from utils import vectorizer_pandas as vp
import pandas as pd
import numpy as np
from utils.const import paths
import os
from utils import merge_csv
from utils.const.stopwords import STOPWORDS
from utils.pre_processor import processor_use_lemma_plus as processor

def clean_and_save_data(df, save_path, save_name):
    """
    Clean text data and save it to a CSV file.
    
    :param df: DataFrame, the data to be cleaned
    :param save_path: str, the directory to save the cleaned data
    :param save_name: str, the name of the saved file
    """
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    file_path = os.path.join(save_path, save_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        
    df.to_csv(file_path, index=False)
    print(f"Cleaned data saved to {file_path}")

def clean_data(read_path=None, file_name=None, save_path=None, save_name='cleaned_merged_training_data.csv', save=True, df=None):
    """
    Clean text data.
    
    :param read_path: str, the directory of the raw data
    :param file_name: str, the name of the raw data file
    :param save_path: str, the directory to save the cleaned data
    :param save_name: str, the name of the saved file
    :param save: bool, whether to save the cleaned data to a CSV file
    :param df: DataFrame, the data to be cleaned (optional)
    :return: DataFrame or None, the cleaned data if save is False, otherwise None
    """
    if df is None:
        if read_path is None:
            read_path = paths.labeled_data
        
        if save_path is None:
            save_path = paths.saved_data_cleaned
        
        if file_name is None:
            df = merge_csv.merge_csv(read_path)
        else:
            try:
                df = pd.read_csv(os.path.join(read_path, file_name))
            except Exception as e:
                print(f"Error reading CSV file: {e}")
                return
    
    df['cleaned_text'] = processor(df['text'].tolist(), STOPWORDS)
    
    if save:
        clean_and_save_data(df, save_path, save_name)
    else:
        return df



def vectorize_and_save(df, vectorizer, vectorizer_type, labels, save_path=None, file_name=None, save=True):
    """
    Vectorize text data and optionally save it to a CSV file.
    
    :param df: DataFrame, the data to be vectorized
    :param vectorizer: the vectorizer to be used
    :param vectorizer_type: str, the type of vectorization ('tf_idf' or 'incidence_matrix')
    :param labels: Series, the labels for the data
    :param save_path: str, the directory to save the vectorized data (optional)
    :param file_name: str, the name of the saved file (optional)
    :param save: bool, whether to save the vectorized data to a CSV file
    :return: DataFrame or None, the vectorized data if save is False, otherwise None
    """
    if vectorizer_type == 'tf_idf':
        result = vp.tf_idf_generator(df, 'cleaned_text', vectorizer)
    elif vectorizer_type == 'incidence_matrix':
        result = vp.incidence_matrix_generator(df, 'cleaned_text', vectorizer)
    else:
        raise ValueError("Invalid vectorizer_type. Must be 'tf_idf' or 'incidence_matrix'.")
    
    result = pd.concat([result, labels], axis=1)
    
    if save:
        if save_path is None or file_name is None:
            raise ValueError("save_path and file_name must be provided when save is True")
        
        file_suffix = f'_{file_name}' if file_name else ''
        file_path = os.path.join(save_path, f'{vectorizer_type}{file_suffix}.csv')
        
        result.to_csv(file_path, index=False)
        print(f"{vectorizer_type} data saved to {file_path}")
    else:
        return result

def vectorize_data(filepath=None, save_path=None, file_name=None, vectorizer=None, vectorizer_type='tf_idf', save=True, df=None):
    """
    Vectorize text data.
    
    :param filepath: str, the path of the cleaned data
    :param save_path: str, the directory to save the vectorized data
    :param file_name: str, the name of the vectorized data file
    :param vectorizer: the vectorizer to be used
    :param vectorizer_type: str, the type of vectorization ('tf_idf' or 'incidence_matrix')
    :param save: bool, whether to save the vectorized data to a CSV file
    :param df: DataFrame, the data to be vectorized (optional)
    :return: DataFrame or None, the vectorized data if save is False, otherwise None
    """
    if df is None:
        if filepath is None:
            filepath = os.path.join(paths.saved_data_cleaned, 'cleaned_merged_training_data.csv')
        
        if not os.path.exists(filepath):
            print("还没有清理好的数据")
            return
        
        df = pd.read_csv(filepath)
    
    
    labels = df['sentiment']
    
    return vectorize_and_save(df, vectorizer, vectorizer_type, labels, save_path, file_name, save)

if __name__ == '__main__':
    clean_data()
    vectorize_data()

# filepath is the path of the cleaned data, save_path is the path to save the vectorized data, file_name is the name of the vectorized data, like 'test_data'
# def clean_data(read_path = None, save_path = None, file_name = None):
#     # 读取数据路径和保存路径
#     if not read_path and not save_path:
#         read_path = paths.labeled_data
#         save_path = paths.saved_data_cleaned
#     if not file_name:
#         file_name = 'cleaned_merged_training_data.csv'
#     else:
#         file_name = f'cleaned_merged_{file_name}.csv'
#     df = merge_csv.merge_csv(read_path)
#     df['cleaned_text'] = processor(df['text'].tolist(), STOPWORDS)
    
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
    
#     if os.path.exists(os.path.join(save_path, file_name)):
#         os.remove(os.path.join(save_path, file_name))
        
#     df.to_csv(os.path.join(save_path, file_name), index=False)    

# filepath is the path of the cleaned data, save_path is the path to save the vectorized data, file_name is the name of the vectorized data, like 'test_data'
# def vectorize_data(filepath = None, save_path = None, file_name = None, vectorizer = None, vectorizer_type = 'tf_idf'):
#     if not filepath:
#         file_path = os.path.join(paths.saved_data_cleaned, 'cleaned_merged_training_data.csv')
    
#     if not save_path:
#         save_path = paths.saved_data_cleaned
    
#     if os.path.exists(file_path):
#         df = pd.read_csv(file_path)
    
#         labels = df['sentiment']
    
#         if vectorizer_type == 'tf_idf':
#             tf_idf = vp.tf_idf_generator(df, 'cleaned_text', vectorizer)
#             tf_idf = pd.concat([tf_idf, labels], axis=1)
#         elif vectorizer_type == 'incidence_matrix':
#             incidence_matrix = vp.incidence_matrix_generator(df, 'cleaned_text', vectorizer)
#             incidence_matrix = pd.concat([incidence_matrix, labels], axis=1)
#         if not file_name:
#             if os.path.exists(os.path.join(save_path, 'incidence_matrix.csv')):
#                 os.remove(os.path.join(save_path, 'incidence_matrix.csv'))
        
#             if os.path.exists(os.path.join(save_path, 'tf_idf.csv')):
#                 os.remove(os.path.join(save_path, 'tf_idf.csv'))
            
#             tf_idf.to_csv(os.path.join(save_path, 'tf_idf.csv'), index=False)
#             incidence_matrix.to_csv(os.path.join(save_path, 'incidence_matrix.csv'), index=False)
#         else:
#             if os.path.exists(os.path.join(save_path, f'{file_name}_incidence_matrix.csv')):
#                 os.remove(os.path.join(save_path, f'{file_name}_incidence_matrix.csv'))
        
#             if os.path.exists(os.path.join(save_path, f'{file_name}_tf_idf.csv')):
#                 os.remove(os.path.join(save_path, f'{file_name}_tf_idf.csv'))
            
#             tf_idf.to_csv(os.path.join(save_path, f'{file_name}_tf_idf.csv'), index=False)
#             incidence_matrix.to_csv((save_path, f'{file_name}_incidence_matrix.csv'), index=False)
#     else:
#         print("还没有清理好的数据")


    