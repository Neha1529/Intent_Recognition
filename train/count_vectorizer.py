#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import CountVectorizer
from joblib import dump, load

def CountVectorizer(train):
    count_vect = CountVectorizer()
    train = count_vect.fit_transform(train)
    dump(count_vect, filename = 'count_vect.joblib')

    return train

    


# In[ ]:




