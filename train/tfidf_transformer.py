#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import TfidfTransformer
from joblib import dump, load

def TfidfTransformer(train):
    count_vect = load('count_vect.joblib')
    train_vect = count_vect.fit_transform(train)
    tfidf_transformer = TfidfTransformer()
    train_tfidf = tfidf_transformer.fit_transform(train_vect)
    dump(tfidf_transformer , filename = 'tfidf_transformer.joblib')
    
    return train_tfidf


# In[ ]:




