import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def read_and_prepare_test_data(file_path):
    test_df = pd.read_csv(file_path)
    test_df.drop(columns=['YEAR', 'FILE_NAME'], inplace=True, errors='ignore')
    test_df.fillna(0, inplace=True)
    return test_df

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R² Score: {r2}")

def load_model_and_test(model_file, X_test_file, y_test_file):
    model = joblib.load(model_file)

    X_test = read_and_prepare_test_data(X_test_file)
    y_test = pd.read_csv(y_test_file, usecols=['2017', '2018', '2019', '2020', '2021', '2022'])

    evaluate_model(model, X_test, y_test)

# Usage
model_file = 'multi_target_lasso_regression_model.joblib'
X_test_file = 'path/to/your/test_data.csv'
y_test_file = 'path/to/your/test_y.csv'

load_model_and_test(model_file, X_test_file, y_test_file)