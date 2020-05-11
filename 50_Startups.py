#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


# In[3]:


startups = pd.read_csv(r"C:\Users\srira\Desktop\Ram\Data science\Course - Assignments\Module 23 - Neural Network\Dataset\50_Startups.csv")
startups.head(10)


# In[5]:


startups = startups.rename(columns={"R&D Spend":"RandD_Spend","Marketing Spend":"Marketing_Spend"})
startups.head(10)


# In[6]:


startups.drop(['State'],axis=1,inplace=True)


# In[7]:


startups.isnull().sum()


# In[9]:


startups.head(10)


# In[10]:


X = startups.drop(['Profit'],axis=1)
Y = startups['Profit']


# In[33]:


x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3)


# In[34]:


X.shape


# In[19]:


#Let's build model


# In[75]:


mlp = MLPRegressor(hidden_layer_sizes=(20,20))


# In[76]:


mlp.fit(x_train,y_train)


# In[77]:


pred_train = mlp.predict(x_train)
pred_test = mlp.predict(x_test)


# In[24]:


#Finding the accuracy i.e. RMSE


# In[78]:


rmse_train = np.sqrt(np.mean(np.square(y_train-pred_train)))
rmse_train


# In[79]:


rmse_test = np.sqrt(np.mean(np.square(y_test-pred_test)))
rmse_test


# In[80]:


#Let's draw plot b/w fitted values vs actual values


# In[81]:


plt.scatter(pred_train,y_train)


# In[ ]:


#So it's a fairly good model

