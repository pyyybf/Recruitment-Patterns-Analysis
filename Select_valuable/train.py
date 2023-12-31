import pandas as pd
import numpy as np
from utils.const import paths
import joblib
import os
from utils.const import paths
from models.random_forest import train_random_forest
from models.SVM import train_SVM
from models.LogisticRegression import train_logistic_regression
from models.DecisionTree import train_decision_tree

def save_model(model, model_name):
    
    filename = os.path.join(paths.saved_models, f"{model_name}.joblib")
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def train_model(model_name, data):
    # 定义模型训练字典
    model_trainers = {
        'RandomForest': train_random_forest,
        'SVM': train_SVM,
        'LogisticRegression': train_logistic_regression,
        'DecisionTree': train_decision_tree
    }
    
    # 检查输入的模型名称是否有效
    if model_name not in model_trainers:
        raise ValueError(f"Invalid model name: {model_name}. Available models: {list(model_trainers.keys())}")
    
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].replace({'Yes': 1, 'No': 0})

    # 调用相应的训练函数
    model = model_trainers[model_name](X, y)
    
    # 保存模型
    save_model(model, model_name)

if __name__ == '__main__':
    if not os.path.exists(paths.saved_models):
        os.makedirs(paths.saved_models)
    if not os.path.exists(os.path.join(paths.saved_data_cleaned, 'tf_idf.csv')):
        raise ValueError("还没有清理好的数据")
    data = pd.read_csv(os.path.join(paths.saved_data_cleaned, 'tf_idf.csv'))
    data = data.dropna()
    train_model(data=data, model_name='RandomForest')