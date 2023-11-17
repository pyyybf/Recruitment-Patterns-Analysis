from utils import vectorizer_pandas as vp
from utils.const import paths
from utils.pre_processor import processor_use_lemma_plus as processor
from utils.const.stopwords import STOPWORDS
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump

def generate_tfidf_matrix(main_folder_path, subfolder_name, encoding='utf-8', save_model=False):
    """
    Reads .txt files from a specified subfolder within a main directory, generates a TF-IDF matrix using these files,
    optionally saves the TfidfVectorizer model, and returns the matrix as a pandas DataFrame.

    :param main_folder_path: Path to the main folder containing subfolders
    :param subfolder_name: Name of the subfolder to process
    :param encoding: Encoding of the .txt files (default is 'utf-8')
    :param save_model: Boolean flag to save the TfidfVectorizer model
    :param model_path: Path to save the TfidfVectorizer model
    :param vectorizer_params: Additional parameters to pass to TfidfVectorizer
    :return: Pandas DataFrame representing the TF-IDF matrix
    """
    print("Converting", subfolder_name, "...")
    subfolder_path = os.path.join(main_folder_path, subfolder_name)

    # Reading the content of each .txt file into a list
    documents = []
    file_names = []
    for file in os.listdir(subfolder_path):
        if file.endswith(".txt"):
            file_path = os.path.join(subfolder_path, file)
            with open(file_path, 'r', encoding=encoding) as file:
                documents.append(file.read())
                file_names.append(file.name)
                file.close()
    documents_cleaned = processor(documents, STOPWORDS)

    # Creating the TF-IDF vectorizer with additional parameters
    vectorizer = TfidfVectorizer(token_pattern=r'\b[a-zA-Z]+\b')
    tfidf_matrix = vectorizer.fit_transform(documents_cleaned)

    # Optionally save the vectorizer model
    if save_model:
        dump(vectorizer, os.path.join(paths.valuable_data, subfolder_name+'tfidf_vectorizer.joblib'))

    # Converting the matrix to a pandas DataFrame
    df = pd.DataFrame(tfidf_matrix.toarray(), index=file_names, columns=vectorizer.get_feature_names_out())
    print('Done!')
    return df



tfidf2016 = generate_tfidf_matrix(paths.valuable_data,"2016",save_model=True)
tfidf2017 = generate_tfidf_matrix(paths.valuable_data,"2017",save_model=True)
tfidf2018 = generate_tfidf_matrix(paths.valuable_data,"2018",save_model=True)
tfidf2019 = generate_tfidf_matrix(paths.valuable_data,"2019",save_model=True)
tfidf2020 = generate_tfidf_matrix(paths.valuable_data,"2020",save_model=True)
tfidf2021 = generate_tfidf_matrix(paths.valuable_data,"2021",save_model=True)
tfidf2022 = generate_tfidf_matrix(paths.valuable_data,"2022",save_model=True)