import pandas as pd
import numpy as np
import joblib
from utils.const import paths
from utils.vectorizer_pandas import tf_idf_generator
from run_clean_vectorize import clean_data, vectorize_data
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import List


def test_model(test_data, model_name):
    model_path = os.path.join(paths.saved_models, f"{model_name}.joblib")
    model = joblib.load(model_path)

    vectorizer_path_tfidf = os.path.join(
        paths.saved_models, 'tfidf_vectorizer.joblib')
    vectorizer = joblib.load(vectorizer_path_tfidf)

    cleaned_df = clean_data(df=test_data, save=False)
    vectorized_df = vectorize_data(
        vectorizer_type='tf_idf', vectorizer=vectorizer, save=False, df=cleaned_df)

    vectorized_df = vectorized_df.dropna()

    # .replace({'Yes': 1, 'No': 0}).astype(int)
    y_test = vectorized_df.iloc[:, -1]
    X_test = vectorized_df.iloc[:, :-1]

    y_pred = model.predict(X_test.values)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")


if __name__ == '__main__':
    test_data = pd.read_csv(os.path.join(
        paths.test_data, 'project-4-at-2023-11-01-05-49-078c8ecf.csv'))
    test_model(test_data, 'SVM')
