#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


data_path = './y_X_sample.csv'
save_model_path = './LinearRegression.joblib'


# In[4]:


data = np.genfromtxt(data_path, delimiter=',')
X_train = data[:, 1:]
y_train = data[:, 0]


# In[5]:


reg = LinearRegression().fit(X_train, y_train)
Rsquared_score = reg.score(X_train, y_train)
y_train_pred = reg.predict(X_train)
mse = mean_squared_error(y_train, y_train_pred)
print(f"Score on Training Set: {Rsquared_score}")
print(f"Mean Squared Error on Training Set: {mse}")


# In[6]:


# save model
joblib.dump(reg, save_model_path)


# In[ ]:




