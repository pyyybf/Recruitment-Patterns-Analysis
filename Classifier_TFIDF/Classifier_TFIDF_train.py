#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import shutil


# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


# In[3]:


import warnings
warnings.filterwarnings("ignore")


# In[4]:


train_data_path = "./y_X_cate_sample.csv"
save_Classifier_TFIDF_train_path = "./Classifier_TFIDF_train"
save_model_path = "./Classifier_TFIDF_train/classification_model"
save_y_train_pred_path = "./Classifier_TFIDF_train/y_train_pred"
output_path = "./Classifier_TFIDF_train/output-classifier.txt"


# In[5]:


data = np.genfromtxt(train_data_path, delimiter=',')
X_train = data[:, 1:]
y_train = data[:, 0]
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)


# In[6]:


models = ['LogisticRegression', 'KNeighborsClassifier', 'SVC', 'DecisionTreeclassifier', 
          'RandomForestclassifier', 'GradientBoostingClassifier', 'NaiveBayes', 'NeuralNetwork']


# In[8]:


def clear_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


# In[9]:

clear_dir(save_Classifier_TFIDF_train_path)
clear_dir(save_model_path)
clear_dir(save_y_train_pred_path)


# In[10]:


def my_print(*values, output_path="./output.txt"):
    values = [str(val) for val in values]
    with open(output_path, "a") as fp:
        fp.write(f"{' '.join(values)}\n")


# In[11]:


def train_classifier(model_name, X_train, y_train):
    if model_name == "LogisticRegression":
        param_grid = {"penalty":['l1', 'l2'],
                      "C": [0.001, 0.01, 0.1, 1, 10],
                      'max_iter': [100, 200, 500]}
        clf = LogisticRegression(solver="sag")
        
    elif model_name == "KNeighborsClassifier":
        param_grid = {'n_neighbors': [1, 3, 5]}
        clf = KNeighborsClassifier()
        
    elif model_name == "SVC":
        param_grid={'C': [0.1, 1, 10]}
        clf = SVC(gamma="auto")
        
    elif model_name == "DecisionTreeclassifier":
        param_grid = {'max_depth': [None, 5, 10, 15], 
                      'min_samples_split': [2, 5, 10]}
        clf = DecisionTreeClassifier()

    elif model_name == "RandomForestclassifier":
        param_grid = {'n_estimators': [100, 500],
                      'max_depth': [None, 5, 10],
                      'min_samples_split': [2, 5, 10],
                      'min_samples_leaf': [1, 2, 4]}
        clf = RandomForestClassifier()
    
    elif model_name == "GradientBoostingClassifier":
        param_grid = {'n_estimators': [100, 500],
                      'learning_rate': [0.01, 0.1, 0.2],
                      'max_depth': [3, 5, 7],
                      'min_samples_split': [2, 5, 10],
                      'min_samples_leaf': [1, 2, 4]}
        clf = GradientBoostingClassifier()

    elif model_name == "NaiveBayes":
        param_grid = {}
        clf = GaussianNB()
        # clf.fit(X_train, y_train)
        # return clf
        
    elif model_name == "NeuralNetwork":
        param_grid = {'hidden_layer_sizes': [(100,), (50, 50), (30, 30, 30)],
                      'alpha': [0.0001, 0.001, 0.01]}
        clf = MLPClassifier()
    
    grid_search = GridSearchCV(clf, param_grid, cv=stratified_kfold, scoring="accuracy", verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


# In[12]:


def evaluate_classifier(y_train_pred, y_train):
    conf_mat = confusion_matrix(y_train_pred, y_train)
    accuracy = accuracy_score(y_train_pred, y_train)
    precision = precision_score(y_train_pred, y_train, average='binary')
    recall = recall_score(y_train_pred, y_train, average='binary')
    f1 = f1_score(y_train_pred, y_train, average='binary')
    
    my_print("Confusion Matrix:", conf_mat, output_path=output_path)
    my_print("Accuracy:", accuracy, output_path=output_path)
    my_print("Precision:", precision, output_path=output_path)
    my_print("Recall:", recall, output_path=output_path)
    my_print("F1 Score:", f1, output_path=output_path)


# In[13]:


for classifier_name in models:
    my_print(f"\n========== {classifier_name} ==========", output_path=output_path)
    # my_print("Start training classifier...", output_path=output_path)
    model = train_classifier(classifier_name, X_train, y_train)
    my_print(model, output_path=output_path)

    # my_print("Start testing classifier...", output_path=output_path)
    y_train_pred = model.predict(X_train)
    evaluate_classifier(y_train_pred, y_train)

    np.savetxt(f'{save_y_train_pred_path}/{classifier_name}_y_pred.csv', y_train_pred, delimiter=',')
    joblib.dump(model, f"{save_model_path}/{classifier_name}.joblib")

