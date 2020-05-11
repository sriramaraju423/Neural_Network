#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor


# In[3]:


concrete = pd.read_csv(r"C:\Users\srira\Desktop\Ram\Data science\Course - Assignments\Module 23 - Neural Network\Dataset\concrete.csv")
concrete.head(10)


# In[4]:


#EDA


# In[5]:


concrete.isnull().sum()


# In[6]:


#Outlier checking


# In[16]:


sb.boxplot(concrete['age'],orient='v')


# In[17]:


# very minimal outliers shouldn't have huge impact. Continuing with modelling


# In[18]:


#Splitting data


# In[20]:


x=concrete.drop(['strength'],axis=1)
y=concrete['strength']


# In[21]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)


# In[33]:


x_train


# In[22]:


#Building model


# In[35]:


mlp = MLPRegressor(hidden_layer_sizes=(15,15))


# In[36]:


mlp.fit(x_train,y_train)


# In[37]:


pred_train = mlp.predict(x_train)


# In[38]:


pred_test = mlp.predict(x_test)


# In[39]:


#Finding RMSE


# In[40]:


rmse_train = np.sqrt(np.mean(np.square(y_train-pred_train)))
rmse_train


# In[41]:


rmse_test = np.sqrt(np.mean(np.square(y_test-pred_test)))
rmse_test


# In[42]:


#Model seems perfect as per RMSE


# In[43]:


#Let's find correlation


# In[45]:


np.corrcoef(pred_train,y_train)


# In[46]:


np.corrcoef(pred_test,y_test)


# In[47]:


#Plot fitted vs observed values


# In[48]:


plt.scatter(pred_train,y_train)


# In[49]:


plt.scatter(pred_test,y_test)


# In[ ]:


# Follows a linear curve, hence model is perfect


# In[8]:


concrete.columns

