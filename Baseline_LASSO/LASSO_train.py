#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, train_test_split
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
alpha = np.concatenate((np.linspace(0.0001, 0.001, 10), np.linspace(0.001, 0.01, 10)))
# tune hyperparameters: alpha
param_grid = {'alpha': alpha}

# GridSearch
grid_search = GridSearchCV(estimator=lasso_model, param_grid=param_grid, 
                           scoring='neg_mean_squared_error', cv=5, verbose=2)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(f"Best Parameters: {best_params}")


# In[6]:


feature_coefficients = best_model.coef_
selected_features = np.where(feature_coefficients != 0)[0]

print("Features with Non-Zero Coefficients:")
print(selected_features)


# In[7]:


y_train_pred = best_model.predict(X_train)
print(y_train_pred)


# In[8]:


mse = mean_squared_error(y_train, y_train_pred)
print(f"Mean Squared Error on Training Set: {mse}")


# In[9]:


# save model
joblib.dump(best_model, save_model_path)


# In[ ]:




