import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def train_random_forest(X, y):
    # 定义模型
    rf = RandomForestClassifier()
    
    # 定义超参数的搜索范围
    param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2', None]
    }
    
    # 使用交叉验证和网格搜索
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X, y)
    
    # 打印最优参数
    print("Best parameters found: ", grid_search.best_params_)
    
    # 返回最优模型
    return grid_search.best_estimator_
