import pandas as pd

def clean_text(corpus):
    
    cleaned_corpus = pd.Series()
    for row in corpus:
        text = []
        for word in row.split():
            p1 = re.sub(pattern='[^a-zA-Z0-9]',repl=' ',string=word)
            p1 = p1.lower()
            text.append(p1)
        cleaned_corpus = cleaned_corpus.append(pd.Series(' '.join(text)))
    
    return cleaned_corpus
cleaned_text = clean_text(complete_corpus)
dump(cleaned_text,'cleaned_text.joblib')
