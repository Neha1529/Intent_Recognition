#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from IntentRecognition.train.count_vectorizer import CountVectorizer
from IntentRecognition.train.tfidf_transformer import TfidfTransformer
from IntentRecognition.train.build_save_model import buildAndSaveModel
from IntentRecognition.train.utils import splitData,createDirectoryForDataStorage

def train_classifier(data):
    df = pd.read_csv(data)
    
    path = createDirectoryForDataStorage()
    
    print('splitting data')
    X_train, X_test, y_train, y_test = splitData(df, y)
    
    print('transforming data')
    X_train_tfidf = TfidfTransformer(X_train)
    
    '''
    Building, Evaluating and Saving Model
    '''
    
    report = buildAndSaveModel(X_train_tfidf, X_test, y_train, y_test, path)
    
    return report


# In[ ]:




