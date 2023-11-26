import pandas as pd
from scipy.sparse import csr_matrix, save_npz
import pandas as pd
import os

os.chdir("C:/Users/tyrza/OneDrive/Documents/USC/Fall2023/ISE540/Project/train_TFIDF/")

csv_files = ['2022_TFIDF.csv',
             '2016_TFIDF.csv', 
             '2017_TFIDF.csv',
             '2018_TFIDF.csv',
             '2019_TFIDF.csv',
             '2020_TFIDF.csv',
             '2021_TFIDF.csv']



def get_cik_from_csv(filename, column_name = 'CIK', chunk_size=1000):
    chunk_iterator = pd.read_csv(filename, usecols=[column_name], chunksize=chunk_size, dtype={column_name: str})
    int_set = set()

    for chunk in chunk_iterator:
        int_set.update(chunk[column_name].dropna().astype(int))

    return int_set


# Initialize the intersection set with the CIK values from the first CSV file
print("Reading 2022 cik")
intersection = get_cik_from_csv(csv_files[0])
print("Got 2022 cik")

# Find the intersection of CIK values in all CSV files
for file in csv_files[1:]:
    print("reading ",file," cik")
    cik_set = get_cik_from_csv(file)
    intersection = intersection.intersection(cik_set)
    print("Got", file,"cik")

# Write the intersection to a file
output_file = "intersection_of_cik_in_training_set.txt"
with open(output_file, "w") as f:
    for cik in intersection:
        f.write(str(cik) + "\n")




