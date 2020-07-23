#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
import datetime
import os

def splitData(df, y):
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=0, stratify=y)
    return X_train, X_test, y_train, y_test


# In[2]:


def createDirectoryForDataStorage():
    ts = datetime.datetime.now().timestamp()
    path = os.path.join(os.getcwd() + "/" + str(int(ts)))
    print("The model and supporting files would be stored in folder named: ", path)
    os.mkdir(path)
    return path


# In[ ]:




