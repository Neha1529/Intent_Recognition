#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request,render_template,redirect,url_for
from joblib import load

from prediction.transformation import transform

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
    
    

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    if request.method == 'POST':
       
        data = request.form['text']
        train_tfidf = transform(data)
    
        prediction = model.predict(train_tfidf)
        output = prediction[0]
        return render_template('index.html', prediction_text='Intent class is  {}'.format(output))

def loadModel():
    classifier = load(r"C:\Users\Neha\IntentRecognition\prediction\model.joblib")
    return classifier

if __name__ == '__main__':
    model = loadModel()
    print("model loading")
    app.run(host = '0.0.0.0',port=5000,debug =True)
    


# In[ ]:



