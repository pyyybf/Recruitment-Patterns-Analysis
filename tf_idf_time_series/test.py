import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import os

os.chdir("C:/Users/tyrza/OneDrive/Documents/USC/Fall2023/ISE540/Project/")

print('Reading y')
y_test = pd.read_csv("test_TFIDF/label.csv")['label']
y_test = y_test.to_list()
print('Reading X')
X_test= pd.read_csv('test_TFIDF/test_2022_TFIDF.csv', header=None)
X_test = X_test.to_numpy()
print('Loading model')
model = joblib.load('train_TFIDF/model/model_LR_time_series.joblib')
print('Predicting')
y_pred = model.predict(X_test)
print('Getting scores...')
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(y_pred)
print("Accuracy:",accuracy)
print("Precision:",precision)
print("Recall",recall)
print("F1-score:",f1)