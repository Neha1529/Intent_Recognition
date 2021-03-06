
#import necessary dependencies and settings

import pandas as pd
import numpy as np
from joblib import dump,load
import logging
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

#Read from file into dataframe
train_url = 'https://raw.githubusercontent.com/Neha1529/IntentRecognition/master/Data/train.csv'
train = pd.read_csv(train_url)
test_url = 'https://raw.githubusercontent.com/Neha1529/IntentRecognition/master/Data/test.csv'
test = pd.read_csv(test_url)
valid_url = 'https://raw.githubusercontent.com/Neha1529/IntentRecognition/master/Data/valid.csv'
valid = pd.read_csv(valid_url)

train = train.append(valid).reset_index(drop=True)

train.shape

train.head()

#Checking for null values in both rows and columns

train.isna().sum().sum()

# Gets or creates a logger
logger = logging.getLogger()

# set log level
logger.setLevel(logging.INFO)

# define file handler and set formatter
file_handler = logging.FileHandler('logfile.log')
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

# add file handler to logger
logger.addHandler(file_handler)

train.append(test).describe()

complete_corpus = pd.Series(train.text.tolist() + test.text.tolist()).astype(str)

logger.info("Preprocessing starts...")
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

complete_corpus = clean_text(complete_corpus)
logger.info("text is cleaned of unnecessary words")

cleaned_text = clean_text(complete_corpus)
dump(cleaned_text,'cleaned_text.joblib')

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

nltk.download('stopwords')
nltk.download('wordnet')

complete_corpus = preprocessing(complete_corpus,remove_stopwords = True,lemmatization=True,remove_accented = True)
logger.info("corpus is cleaned of stop words,accented words")
logger.info("preprocessing ends here")

preprocessed_data = preprocessing(complete_corpus,remove_stopwords = True,lemmatization=True,remove_accented = True)
dump(preprocessed_data, 'preprocessing.joblib')

train_corpus = complete_corpus[0:train.shape[0]]
test_corpus = complete_corpus[train.shape[0]:]

#before preprocessing
train.text[10]

#After text preprocessing
train_corpus[10]

tfidf = TfidfVectorizer()
features = tfidf.fit_transform(train_corpus).toarray()
features.shape
dump(tfidf , 'tfidf.joblib')
logger.info("train corpus vectorized")

"""# Train-Test split"""

X_train, X_test, y_train, y_test = train_test_split(train_corpus, train.intent, test_size = 0.2, random_state = 0)

#instantiate CountVectorizer()
count_vect = CountVectorizer()
    
#generates word counts for the words in your doc
X_train_counts = count_vect.fit_transform(X_train)
dump(count_vect , 'count_vect.joblib')
logger.info("Document-term matrix returned by count vectorizer")
#instantiate TfidfTransformer()
tfidf_transformer = TfidfTransformer()
    
#gets the tf-idf score for the document
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
dump(tfidf_transformer , 'tfidf_transformer.joblib')
logger.info("tfidf score returned by tfidf_transformer")

X_train_tfidf1 = tfidf_transformer.fit(count_vect.fit_transform(X_train))
#print idf values
df_idf = pd.DataFrame(X_train_tfidf1.idf_, index=count_vect.get_feature_names(),columns=["idf_weights"])

#sort ascending
df_idf.sort_values(by=['idf_weights']).head()

model = RandomForestClassifier(n_estimators=50,criterion = 'entropy', random_state=0)
logging.info("model is created")
clf = model.fit(X_train_tfidf, y_train)
logging.info("model is fitted on train corpus")
y_pred = clf.predict(count_vect.transform(X_test))
logger.info("model transformed for test corpus")

print(clf.predict(count_vect.transform(["find me a table for two"])))

print(clf.predict(count_vect.transform(["will it rain in Delhi"])))

print(clf.predict(count_vect.transform(["search for the adventures of TinTin"])))

logger.info("Start cross validation")
cv=5
labels = train.intent

#Evaluating a score by cross-validation
accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=cv)


#Array of scores of the estimator for each run of the cross validation
print("Accuracy for the folds:",accuracies)

score = np.mean(accuracies)
print(score)
logger.info("The mean score for model :{}".format(score))
logger.info("Cross validation ends here")

from sklearn import metrics
print(metrics.classification_report(y_test, y_pred))

print("Precision score:",metrics.precision_score(y_test, y_pred ,labels = train['intent'],average='weighted'))

"""#Model Serialization"""

import pickle
filehandler = open("model.pkl","wb")
pickle.dump(model,filehandler)
filehandler.close()
