import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import joblib
from utils.const import paths
import os

def incidence_matrix_generator(data, column_name, vectorizer = None):
    """
    This function generates an incidence matrix for NLP data.
    
    Parameters:
        data (pd.DataFrame): A dataframe containing the text data.
        column_name (str): The name of the column in the dataframe that contains the text data.
    
    Returns:
        pd.DataFrame: Incidence matrix
    """
    data[column_name] = data[column_name].fillna('')
    
    token_pattern = r'\b[a-zA-Z]+\b'
    
    if not vectorizer:
        vectorizer = CountVectorizer(binary=True, token_pattern=token_pattern)
    
        incidence_matrix_vectorizer = vectorizer.fit(data[column_name])
        joblib.dump(incidence_matrix_vectorizer, os.path.join(paths.saved_models, 'incidence_matrix_vectorizer.joblib'))
    else:
        incidence_matrix_vectorizer = vectorizer
    
    X = incidence_matrix_vectorizer.transform(data[column_name])
    
    incidence_matrix = pd.DataFrame(X.toarray(), columns=incidence_matrix_vectorizer.get_feature_names_out())
    return incidence_matrix

def tf_idf_generator(data, column_name, vectorizer = None):
    """
    This function takes in a dataframe and a column name, 
    and returns a dataframe with the TF-IDF values for each word in the column.
    It also removes words that are entirely numeric.
    
    Parameters:
        data (pd.DataFrame): The input dataframe.
        column_name (str): The name of the column to calculate TF-IDF values for.
        
    Returns:
        pd.DataFrame: A dataframe with the TF-IDF values.
    """
    # Handle missing values
    data[column_name] = data[column_name].fillna('')
    
    # Create the TF-IDF vectorizer object
    # The token pattern here ensures that words are made of letters (both uppercase and lowercase) only
    
    if not vectorizer:
        vectorizer = TfidfVectorizer(token_pattern=r'\b[a-zA-Z]+\b')
    
        # Fit the TF-IDF vectorizer and transform the data
        tfidf_vectorizer = vectorizer.fit(data[column_name])
        joblib.dump(tfidf_vectorizer, os.path.join(paths.saved_models, 'tfidf_vectorizer.joblib'))
    else:
        tfidf_vectorizer = vectorizer
    
    tfidf_vectors = tfidf_vectorizer.transform(data[column_name])
    
    # Create a dataframe with the TF-IDF vectors
    tfidf_df = pd.DataFrame(tfidf_vectors.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=data.index)
    
    # Return the dataframe
    return tfidf_df


