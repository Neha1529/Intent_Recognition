#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import necessary dependencies and settings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rc
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
from pylab import rcParams
import seaborn as sns

import re,nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import unicodedata
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from joblib import dump, load


# In[2]:


#Read from file into dataframe

train = pd.read_csv("downloads/Data/train.csv")
test = pd.read_csv("downloads/Data/test.csv")
valid = pd.read_csv("downloads/Data/valid.csv")

train = train.append(valid).reset_index(drop=True)


# In[3]:


train.shape


# In[4]:


train.head()


# In[5]:


#Checking for null values in both rows and columns

train.isna().sum().sum()


# In[6]:


#Checking number of texts per intent

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8


# In[7]:



chart = sns.countplot(train.intent, palette=COLORS_PALETTE)
plt.title("Number of texts per intent")
chart.set_xticklabels(chart.get_xticklabels(), rotation=30, horizontalalignment='right');
plt.show()


# In[8]:


train.append(test).describe()


# In[9]:


complete_corpus = pd.Series(train.text.tolist() + test.text.tolist()).astype(str)


# In[10]:


def clean_text(corpus):
    '''
    Purpose : Function to keep only alphabets, digits and certain words (punctuations, qmarks, tabs etc. removed)
    
    Input : Takes a text corpus, 'corpus' to be cleaned along with a list of words, 
    
    Output : Returns the cleaned text corpus
    
    '''
    cleaned_corpus = pd.Series()
    for row in corpus:
        text = []
        for word in row.split():
            p1 = re.sub(pattern='[^a-zA-Z0-9]',repl=' ',string=word)
            p1 = p1.lower()
            text.append(p1)
        cleaned_corpus = cleaned_corpus.append(pd.Series(' '.join(text)))
    return cleaned_corpus


# In[11]:


complete_corpus = clean_text(complete_corpus)


# In[12]:


#Removing stopwords,accented words and lemmatizing

def preprocessing(corpus, remove_stopwords=True, lemmatization=True, remove_accented=True):
    if remove_stopwords == True:
        wh_words = ['who', 'what', 'when', 'why', 'how', 'which', 'where', 'whom']
        stop = set(stopwords.words('english'))
        for word in wh_words:
            stop.remove(word)
        corpus = [[x for x in x.split() if x not in stop] for x in corpus]
    else :
        corpus = [[x for x in x.split()] for x in corpus]
    
    if lemmatization == True:
        lemmatizer = WordNetLemmatizer()
        corpus = [[lemmatizer.lemmatize(x, pos = 'v') for x in x] for x in corpus]
        
    if remove_accented == True:
        corpus = [[unicodedata.normalize('NFKD',x).encode('ascii', 'ignore').decode('utf-8', 'ignore')for x in x]for x in corpus]
    
    corpus = [' '.join(x) for x in corpus]
    return corpus   


# In[13]:


complete_corpus = preprocessing(complete_corpus,remove_stopwords = True,lemmatization=True,remove_accented = True)


# In[14]:


train_corpus = complete_corpus[0:train.shape[0]]
test_corpus = complete_corpus[train.shape[0]:]


# In[15]:


#before preprocessing
train.text[10]


# In[16]:


#After text preprocessing
train_corpus[10]


# In[17]:


tfidf = TfidfVectorizer()
features = tfidf.fit_transform(train_corpus).toarray()
features.shape
dump(tfidf , 'tfidf.joblib')


# # Train-Test split

# In[18]:


X_train, X_test, y_train, y_test = train_test_split(train_corpus, train.intent, test_size = 0.2, random_state = 0)


# In[19]:


#instantiate CountVectorizer()
count_vect = CountVectorizer()
    
#generates word counts for the words in your doc
X_train_counts = count_vect.fit_transform(X_train)

dump(count_vect , 'count_vect.joblib')


# In[20]:


#instantiate TfidfTransformer()
tfidf_transformer = TfidfTransformer()
    
#gets the word counts for the documents in a sparse matrix form
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

dump(tfidf_transformer , 'tfidf_transformer.joblib')


# In[21]:


X_train_tfidf1 = tfidf_transformer.fit(count_vect.fit_transform(X_train))
#print idf values
df_idf = pd.DataFrame(X_train_tfidf1.idf_, index=count_vect.get_feature_names(),columns=["idf_weights"])

#sort ascending
df_idf.sort_values(by=['idf_weights']).head()


# In[22]:


model = RandomForestClassifier(n_estimators=50, criterion = 'entropy', random_state=0)
clf = model.fit(X_train_tfidf, y_train)
y_pred = clf.predict(count_vect.transform(X_test))


# In[23]:


print(clf.predict(count_vect.transform(["find me a table for two"])))


# In[24]:


print(clf.predict(count_vect.transform(["will it rain in Delhi"])))


# In[25]:


print(clf.predict(count_vect.transform(["search for the adventures of Gulliver"])))


# In[26]:


cv=5
labels = train.intent

#Evaluating a score by cross-validation
accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=cv)

#Array of scores of the estimator for each run of the cross validation
print("Accuracy for the folds:",accuracies)


# In[27]:


import numpy as np
print(np.mean(accuracies))


# In[28]:


print(metrics.classification_report(y_test, y_pred))


# In[29]:


print("Precision score:",metrics.precision_score(y_test, y_pred ,labels = train['intent'],average='weighted'))


# #Model Serialization

# In[30]:


dump(model , 'model.joblib')


# In[ ]:




