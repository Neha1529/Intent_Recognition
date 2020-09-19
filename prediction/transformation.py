
from joblib import load
from preprocessing.preprocessing import preprocessing


def load_count_vectorizer():
    count_vect = load(r'/home/site/wwwroot/prediction/count_vect.joblib')
    return count_vect


# In[3]:


def load_tfidf_transformer():
    tfidf_transformer = load(r'/home/site/wwwroot/prediction/tfidf_transformer.joblib')
    return tfidf_transformer


# In[4]:


def transform(data):
    count_vect = load_count_vectorizer()
    tfidf_transformer = load_tfidf_transformer()
    preprocessed_data = preprocessing([data],remove_stopwords=True, lemmatization=True, remove_accented=True)
    train_cv = count_vect.transform(preprocessed_data)
    X_train_idf = tfidf_transformer.transform(train_cv)
    
    return X_train_idf


# In[ ]:




