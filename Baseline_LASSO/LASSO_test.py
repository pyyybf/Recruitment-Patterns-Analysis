#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.metrics import mean_squared_error
import numpy as np
import joblib


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


data_path = './y_X_sample.csv'
load_model_path = './Best_LASSO.joblib'


# In[4]:


data = np.genfromtxt(data_path, delimiter=',')
X_test = data[:, 1:]
y_test = data[:, 0]


# In[5]:


# load model
best_model = joblib.load(load_model_path)


# In[6]:


y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse}")


# In[7]:


# save the prediction values - 2D NumPy array to a file
np.savetxt('y_pred.csv', y_pred, delimiter=',')


# In[ ]:




