#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np


# In[25]:


df=pd.read_csv(r"D:\ai\g.h.I\CarBuyer.csv")


# In[26]:


df.head(2)


# In[27]:


y = df[['carBuyer']]
x = df.drop(['carBuyer','EnglishEducation'],axis=1)
x=x.dropna()


# In[28]:


x.head(2)


# In[29]:


y.head(2)


# In[30]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)


# In[31]:


x_train.shape


# In[32]:


y_train.shape


# In[33]:


from sklearn.neural_network import MLPClassifier


# In[34]:


anna = MLPClassifier(max_iter=500,activation='relu',hidden_layer_sizes=(2, 2))


# In[35]:


anna


# In[36]:


anna.fit(x_train,y_train)


# In[37]:


pred=anna.predict(x_test)
pred


# In[38]:


from sklearn.metrics import classification_report


# In[40]:


classification_report(y_test,pred)


# In[41]:


from sklearn.metrics import confusion_matrix


# In[42]:


confusion_matrix(y_test,pred)


# In[ ]:




