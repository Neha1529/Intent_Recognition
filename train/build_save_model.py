#!/usr/bin/env python
# coding: utf-8

# In[1]:


from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def buildAndSaveModel(X_train, X_test, y_train, y_test, path):
    print("Building model")
    model = RandomForestClassifier(n_estimators=200, max_depth= 5, random_state=42)
    model.fit(X_train, y_train)
    print("Validating model")
    # Predicting the Test set results
    y_pred = model.predict(X_test)

    report = metrics.classification_report(y_test, y_pred)
    print('classification report', report)

    dump(model, filename = path + "/" + 'model.joblib')

    return report


# In[ ]:




