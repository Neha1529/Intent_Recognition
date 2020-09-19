
import re,nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import unicodedata

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
