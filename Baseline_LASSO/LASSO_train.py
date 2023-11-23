#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


data_path = './y_X_sample.csv'
save_model_path = './Best_LASSO.joblib'


# In[4]:


data = np.genfromtxt(data_path, delimiter=',')
X_train = data[:, 1:]
y_train = data[:, 0]


# In[5]:


lasso_model = Lasso()

# tune hyperparameters: alpha
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100, 1000]}

# GridSearch
grid_search = GridSearchCV(estimator=lasso_model, param_grid=param_grid, 
                           scoring='neg_mean_squared_error', cv=5, verbose=2)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(f"Best Parameters: {best_params}")


# In[6]:


# save model
joblib.dump(best_model, save_model_path)

