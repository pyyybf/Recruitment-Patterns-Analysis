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


# training data
data_path = 'y_X_sample.csv'
load_model_path = 'Best_LASSO.joblib'


# In[4]:


# load model
best_model = joblib.load(load_model_path)


# In[5]:


feature_coefficients = best_model.coef_

selected_features = np.where(feature_coefficients != 0)[0]

print("Features with Non-Zero Coefficients:")
print(selected_features)


# In[6]:


data = np.genfromtxt(data_path, delimiter=',')
X_train = data[:, 1:]
y_train = data[:, 0]


# In[7]:


y_train_pred = best_model.predict(X_train)
mse = mean_squared_error(y_train, y_train_pred)
print(f"Mean Squared Error on Training Set: {mse}")


# In[ ]:




