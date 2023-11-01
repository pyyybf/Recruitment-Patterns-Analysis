from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def train_SVM(X, y):
    # 定义模型
    svc = SVC()
    
    # 定义超参数的搜索范围
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
    }
    
    # 使用交叉验证和网格搜索
    grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X, y)
    
    # 打印最优参数
    print("Best parameters found: ", grid_search.best_params_)
    
    # 返回最优模型
    return grid_search.best_estimator_
