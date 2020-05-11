#!/usr/bin/env python
# coding: utf-8

# In[84]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import seaborn as sb


# In[70]:


fireforests = pd.read_csv(r"C:\Users\srira\Desktop\Ram\Data science\Course - Assignments\Module 23 - Neural Network\Dataset\fireforests.csv")
fireforests.head(10)


# In[35]:


#It doensn't matter the day of the incident for modelling


# In[4]:


#Output area to be considered as categorical which implies area burned is small or large


# In[71]:


fireforests['area'].value_counts()


# In[6]:


#Converting data to categorical


# In[72]:


bins=[0,10,1100]
fireforests['area'] = pd.cut(fireforests['area'],bins,labels=['0','1'],include_lowest=True)


# In[73]:


fireforests['area'].value_counts()


# In[74]:


fireforests = fireforests.iloc[:,2:11]
fireforests.head(10)


# In[38]:


#EDA


# In[39]:


sb.pairplot(fireforests.iloc[:,:])


# In[40]:


#Correlation is not that great


# In[75]:


fireforests.isnull().sum()


# In[42]:


#Handling outliers


# In[76]:


sb.boxplot(fireforests['rain'],orient='v')


# In[44]:


# Outlier columns - FFMC,DMC,DC,ISI,temp,RH,rain


# In[77]:


fireforests_outlier_df = fireforests.drop(['area'],axis=1)


# In[78]:


Q1 = fireforests_outlier_df.quantile(0.25)
Q3 = fireforests_outlier_df.quantile(0.75)
IQR = Q3 - Q1
IQR


# In[79]:


for i in fireforests_outlier_df.columns:
    min_value = fireforests_outlier_df[i].min()
    max_value = fireforests_outlier_df[i].max()
    fireforests_outlier_df[i]=np.where(fireforests_outlier_df[i]<(Q1-1.5*IQR)[i],min_value,fireforests_outlier_df[i])
    fireforests_outlier_df[i]=np.where(fireforests_outlier_df[i]>(Q3+1.5*IQR)[i],max_value,fireforests_outlier_df[i])
fireforests_outlier_df


# In[80]:


sb.boxplot(fireforests_outlier_df['DMC'],orient='v')


# In[81]:


fireforests.columns


# In[82]:


# x = fireforests.drop(['area'],axis=1)
# y = fireforests['area']
x = fireforests_outlier_df
y = fireforests['area']


# In[51]:


#Splitting the data into train and test


# In[83]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)


# In[53]:


#Regression doesn't need standardization


# In[54]:


#building model


# In[85]:


mlp=MLPClassifier(hidden_layer_sizes=(20,20))


# In[86]:


mlp.fit(x_train,y_train)


# In[87]:


pred_train = mlp.predict(x_train)


# In[88]:


pred_test = mlp.predict(x_test)


# In[90]:


#Checking accuracy


# In[91]:


acc_train = np.mean(y_train==pred_train)
acc_test = np.mean(y_test==pred_test)
acc_train,acc_test

