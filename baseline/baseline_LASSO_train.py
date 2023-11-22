import pandas as pd
import os
from sklearn.linear_model import MultiTaskLasso
import joblib

def read_and_clean_csv(file_path):
    df = pd.read_csv(file_path)
    df.drop(columns=['YEAR', 'FILE_NAME'], inplace=True, errors='ignore')
    return df

def merge_and_sum_dataframes(dataframes):
    merged_df = pd.concat(dataframes)
    merged_df.fillna(0, inplace=True)
    return merged_df.groupby('CIK').sum()

def process_files_and_train_multi_target_model(folder_path, years, y_train_file, model_file):
    dataframes = [read_and_clean_csv(os.path.join(folder_path, f"{year}_TFIDF.csv"))
                  for year in years if os.path.exists(os.path.join(folder_path, f"{year}_TFIDF.csv"))]
    
    final_df = merge_and_sum_dataframes(dataframes)
    y_train = pd.read_csv(y_train_file, usecols=['cik', '2017', '2018', '2019', '2020', '2021', '2022'])

    # Ensure the target array is aligned with the DataFrame
    final_df = final_df[final_df.index.isin(y_train['cik'])]
    y_train = y_train[y_train['cik'].isin(final_df.index)].set_index('cik')

    # Train Multi-Target Lasso Regression Model
    multi_lasso_reg = MultiTaskLasso(alpha=1.0, max_iter=1000)
    multi_lasso_reg.fit(final_df, y_train[['2017', '2018', '2019', '2020', '2021', '2022']])

    # Save the model
    joblib.dump(multi_lasso_reg, model_file)
    print(f"Multi-target model saved as {model_file}")

# Usage
folder_path = 'path/to/your/csv/files'
years = ["2016", "2017", "2018", "2019", "2020", "2021", "2022"]
y_train_file = 'train_y.csv'
model_file = 'multi_target_lasso_regression_model.joblib'

process_files_and_train_multi_target_model(folder_path, years, y_train_file, model_file)

