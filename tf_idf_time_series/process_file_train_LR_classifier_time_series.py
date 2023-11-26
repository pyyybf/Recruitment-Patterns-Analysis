import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import joblib

os.chdir("C:/Users/tyrza/OneDrive/Documents/USC/Fall2023/ISE540/Project/train_TFIDF/")

y = pd.read_csv("train_label/label_2022.csv")['label_2022']
y = y.to_list()

def csv_to_numpy_without_column(csv_file_path, column_to_delete="CIK"):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Delete the specified column
    if column_to_delete in df.columns:
        df.drop(column_to_delete, axis=1, inplace=True)
    else:
        raise ValueError(f"Column '{column_to_delete}' not found in the CSV file.")

    # Convert the DataFrame to a NumPy array
    array = df.to_numpy()
    np.save(f"array_{csv_file_path.replace('.csv', '.npy')}", array)

# Usage
column_to_delete = 'CIK'
print("Reading 2016")
csv_to_numpy_without_column('filtered_matched_2016_TFIDF.csv')
print("Reading 2017")
csv_to_numpy_without_column('filtered_matched_2017_TFIDF.csv')
print("Reading 2018")
csv_to_numpy_without_column('filtered_matched_2018_TFIDF.csv')
print("Reading 2019")
csv_to_numpy_without_column('filtered_matched_2019_TFIDF.csv')
print("Reading 2020")
csv_to_numpy_without_column('filtered_matched_2020_TFIDF.csv')
print("Reading 2021")
csv_to_numpy_without_column('filtered_matched_2021_TFIDF.csv')
print("Reading 2022")
csv_to_numpy_without_column('filtered_matched_2022_TFIDF.csv')

tfidf2016 = np.load('array_filtered_matched_2016_TFIDF.npy')
tfidf2017 = np.load('array_filtered_matched_2017_TFIDF.npy')
tfidf2018 = np.load('array_filtered_matched_2018_TFIDF.npy')
tfidf2019 = np.load('array_filtered_matched_2019_TFIDF.npy')
tfidf2020 = np.load('array_filtered_matched_2020_TFIDF.npy')
tfidf2021 = np.load('array_filtered_matched_2021_TFIDF.npy')
tfidf2022 = np.load('array_filtered_matched_2022_TFIDF.npy')

parameter_list = [0.5,0.6,0.7,0.8,0.9]
best_parameter = 0
best_mean_cv_score = 0


def get_X(parameter):
    X = ((parameter**6) * tfidf2016 + 
         (parameter**5) * tfidf2017 + 
         (parameter**4) * tfidf2018 +
         (parameter**3) * tfidf2019 + 
         (parameter**2) * tfidf2020 + 
         (parameter) * tfidf2021 + tfidf2022)/((parameter**6)+
                                               (parameter**5)+
                                               (parameter**4)+
                                               (parameter**3)+
                                               (parameter**2)+
                                               (parameter**1)+1)
    return X

for parameter in parameter_list:
    print("Running model with",parameter,"as parameter")
    X = get_X(parameter)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(C=1,solver='sag',penalty='l2', max_iter=1000,random_state=123)
    cv_scores = cross_val_score(model, X, y, cv=5,scoring='accuracy')
    print(cv_scores)
    print("Mean CV score is",cv_scores.mean())
    
    if cv_scores.mean() > best_mean_cv_score:
        best_mean_cv_score = cv_scores.mean()
        best_parameter = parameter
        joblib.dump(model, 'model/LR.joblib')
