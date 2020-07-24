
from joblib import load


def load_count_vectorizer():
    count_vect = load(r'/home/wwwroot/prediction/count_vect.joblib')
    return count_vect


# In[3]:


def load_tfidf_transformer():
    tfidf_transformer = load(r'/home/site/wwwroot/prediction/tfidf_transformer.joblib')
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




