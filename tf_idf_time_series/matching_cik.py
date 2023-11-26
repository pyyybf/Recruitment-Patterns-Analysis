import pandas as pd
from scipy.sparse import csr_matrix, save_npz
import pandas as pd
import os

os.chdir("C:/Users/tyrza/OneDrive/Documents/USC/Fall2023/ISE540/Project/train_TFIDF/")

# processed_csv_files = ['processed_2022_TFIDF.csv',
#              'processed_2016_TFIDF.csv', 
#              'processed_2017_TFIDF.csv',
#              'processed_2018_TFIDF.csv',
#              'processed_2019_TFIDF.csv',
#              'processed_2020_TFIDF.csv',
#              'processed_2021_TFIDF.csv']

csv_files = ['2022_TFIDF.csv',
             '2016_TFIDF.csv', 
             '2017_TFIDF.csv',
             '2018_TFIDF.csv',
             '2019_TFIDF.csv',
             '2020_TFIDF.csv',
             '2021_TFIDF.csv']

def read_file_to_set(file_path):
    with open(file_path, 'r') as file:
        return {int(line.strip()) for line in file}
    

def match_intersect_cik(file_path, output_file_path, cik_set, chunk_size=500):
    chunk_iterator = pd.read_csv(file_path, chunksize=chunk_size)
    first_chunk = True

    for chunk in chunk_iterator:
        # Drop unwanted columns
        chunk.dropna(how='all', inplace=True)
        chunk.drop(columns=['YEAR', 'FILE_NAME'], inplace=True, errors='ignore')

        chunk['CIK'] = chunk['CIK'].astype(int)
        chunk = chunk[chunk['CIK'].isin(cik_set)]

        # Write the processed chunk to the output file
        if first_chunk:
            chunk.to_csv(output_file_path, index=False)
            first_chunk = False
        else:
            chunk.to_csv(output_file_path, mode='a', header=False, index=False)

# def process_and_save_csv(file_path, output_file_path, column_name="CIK", chunk_size=500):
#     chunk_iterator = pd.read_csv(file_path, chunksize=chunk_size)
#     first_chunk = True

#     for chunk in chunk_iterator:
#         chunk.dropna(how='all', inplace=True)
#         chunk[column_name] = chunk[column_name].astype(int)

#         if first_chunk:
#             chunk.to_csv(output_file_path, index=False)
#             first_chunk = False
#         else:
#             chunk.to_csv(output_file_path, mode='a', header=False, index=False)


cik_set = read_file_to_set('intersection_of_cik_in_training_set.txt')

for file in csv_files:
    print("Processing",file)
    match_intersect_cik(file, f"matched_{file}", cik_set)
    print("Done with",file)