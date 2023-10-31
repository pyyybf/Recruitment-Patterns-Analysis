import dask.dataframe as dd
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import dask.array as da
from scipy import sparse

def dask_incidence_matrix_generator(data, column_name):
    """
    This function generates a dask dataframe representing an incidence matrix for NLP data.
    
    Parameters:
        data (dd.DataFrame): A dask dataframe containing the text data.
        column_name (str): The name of the column in the dataframe that contains the text data.
    
    Returns:
        dd.DataFrame: Incidence matrix as a dask dataframe
    """
    token_pattern = r'\b[a-zA-Z]+\b'
    vectorizer = CountVectorizer(binary=True, token_pattern=token_pattern)
    
    # Compute to bring data into memory for vectorization
    data = data.compute()
    X = vectorizer.fit_transform(data[column_name])
    
    # Convert the sparse matrix to a dask array
    dask_array = da.from_array(X, asarray=False, fancy=False, chunks=1000)
    incidence_matrix = dd.from_dask_array(dask_array, columns=vectorizer.get_feature_names_out())
    
    return incidence_matrix

def dask_tf_idf_generator(data, column_name):
    """
    This function takes in a dask dataframe and a column name, 
    and returns a dask dataframe with the TF-IDF values for each word in the column.
    
    Parameters:
        data (dd.DataFrame): The input dask dataframe.
        column_name (str): The name of the column to calculate TF-IDF values for.
        
    Returns:
        dd.DataFrame: A dask dataframe with the TF-IDF values.
    """
    # Handle missing values
    data[column_name] = data[column_name].fillna('')
    
    # Compute to bring data into memory for vectorization
    data = data.compute()
    
    # Create the TF-IDF vectorizer object
    tfidf = TfidfVectorizer(token_pattern=r'\b[a-zA-Z]+\b')
    
    # Fit the TF-IDF vectorizer and transform the data
    tfidf_vectors = tfidf.fit_transform(data[column_name])
    
    # Convert the sparse matrix to a dask array
    dask_array = da.from_array(tfidf_vectors, asarray=False, fancy=False, chunks=1000)
    tfidf_df = dd.from_dask_array(dask_array, columns=tfidf.get_feature_names_out())
    
    return tfidf_df
