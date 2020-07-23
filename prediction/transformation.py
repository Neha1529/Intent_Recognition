#!/usr/bin/env python
# coding: utf-8

# In[1]:


from joblib import load


# In[2]:


def load_count_vectorizer():
    count_vect = load(r'C:\Users\Neha\IntentRecognition\prediction\count_vect.joblib')
    return count_vect


# In[3]:


def load_tfidf_transformer():
    tfidf_transformer = load(r'C:\Users\Neha\IntentRecognition\prediction\tfidf_transformer.joblib')
    return tfidf_transformer


# In[4]:


def transform(data):
    count_vect = load_count_vectorizer()
    tfidf_transformer = load_tfidf_transformer()
    print([data])
    train_cv = count_vect.transform([data])
    print(train_cv)
    X_train_idf = tfidf_transformer.transform(train_cv)
    
    return X_train_idf


# In[ ]:




