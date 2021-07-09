# Imports: Utility
import re
import requests
import json
import string
import numpy as np
import pickle

# Flask / DB / Scraping
from bs4 import BeautifulSoup
from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy

# Deep learning
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences

# Instance of Flask App
app = Flask(__name__)

# Local Database connection
app.config['SQLALCHEMY_DATABASE_URI']='postgresql://dhrumilp:test123@localhost/datapassports'

# DB Instance and model
# Can be further optimized by creating models.py file to keep all models separate
db = SQLAlchemy(app)
class Data(db.Model):
    
    # Create a table
    __tablename__ = "data"
    id = db.Column(db.Integer, primary_key = True)
    name = db.Column(db.String)
    org = db.Column(db.String)
    link = db.Column(db.String)

    # Constructor
    def __init__(self, name, org, link):
        self.name = name
        self.org = org
        self.link = link

# Home-page
@app.route('/', methods = ['GET','POST'])
def ner():
    """
    Renders home page to receive link/text from user.
    """
    return render_template('index.html')

# Results-page
@app.route('/result', methods = ['GET', 'POST'])
def result():
    """
    Takes link/text from user, scrapes data, predicts results and renders result page with results
    """
    if request.method == 'POST':
      link = request.form['link']

    # Scraping using BeautifulSoup
    page = requests.get(link)
    soup = BeautifulSoup(page.content,"html.parser")
    article = soup.find("div",{"class":"wysiwyg wysiwyg--all-content css-1vsenwb"}).text
    # article = "Barak Obama was the president of the United States of America while he met with Bill Gates who founded Google when he was only 17"

    # Cleaning
    words = article.split()
    words.append("ENDPAD")
    num_words = len(words)
    sentences = article.split('.')

    # Word indexing, returns following:
    # word2idx = {
    #     'obama': 1,
    #     'was': 5,
    #     'president': 8,
    # }
    word2idx = pickle.load(open('../model/word_to_index.pickle', 'rb'))
    tags2idx = pickle.load(open('../model/tag_to_index.pickle', 'rb'))

    # Helper for test data
    def get_test_data(article, words, word2idx, sentences, num_words):
        clean_sentences = []
        tokens = []
        for element in sentences:
            clean_sentences.append(element.strip())
            tokens.append(element.split())

        # Sentence slicing/indexing
        # Returns [[1,4,6,9],[230,44,552],...,n]
        X_test = []
        for s in range(len(tokens)):
            temp = []
            for w in range(len(tokens[s])):
                if tokens[s][w] in word2idx:
                    temp.append(word2idx[tokens[s][w]])
            X_test.append(temp)

        # Adding padding
        x_test = pad_sequences(sequences=X_test,
                            padding="post", value=num_words+1, maxlen=50)
        
        return x_test
    
    # Get test data and convert to numpy array for prediction
    X_test = get_test_data(article, words, word2idx, sentences, num_words)
    x_test = np.asarray(X_test) 

    # Load Model and Predict
    model = keras.models.load_model('../model/ner_model.h5')
    tags = [u'I-art', u'B-gpe', u'B-art', u'I-per', u'I-tim', u'B-org', u'O', u'B-geo', u'B-tim', u'I-geo', u'B-per', u'I-eve', u'B-eve', u'I-gpe', u'I-org', u'I-nat', u'B-nat']
    
    i = np.random.randint(0, x_test.shape[0])
    p = model.predict(np.array([x_test[i]]))
    p = np.argmax(p, axis=-1)
    
    # Return Name and Org after prediction
    name = []
    org = []
    for w, pred in zip(x_test[i], p[0]):
        word = list(word2idx.keys())[list(word2idx.values()).index(w)]
        tag = list(tags2idx.keys())[list(tags2idx.values()).index(pred)]
        if tags[pred] == "I-per":
            name.append(word)
        elif tags[pred] == "I-org":
            org.append(word)
        
    data = Data(name, org, link)
    db.session.add(data)
    db.session.commit()

    # Data to be passed to template to render
    data = {
        'link' : link,
        'article' : article,
        'test' : x_test,
        'pred' : p[0],
        'word2idx' : word2idx,
        'name' : name,
        'org' : org,
    }

    return render_template('result.html', data=data)

if __name__ == '__main__':
   app.run(debug = True)