import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, Response, request, url_for, redirect
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer


app = Flask(__name__)
ps = PorterStemmer()


def transform_text(text):
    aman = ""
    for i in text:
        aman = aman+" "+text[0]
    aman = aman.lower()
    aman = nltk.word_tokenize(aman)

    y = []
    for i in aman:
        if i.isalnum():
            y.append(i)

    aman = y[:]
    y.clear()

    for i in aman:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    aman = y[:]
    y.clear()

    for i in aman:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('amanvector.pkl', 'rb'))
model = pickle.load(open('amanmodel.pkl', 'rb'))


def fake_news_det(news):
    input_data = [news]
    transform_news = transform_text(input_data)
    vectorized_news = tfidf.transform([transform_news])
    prediction = model.predict(vectorized_news)[0]
    if prediction == 1:
        return ['REAL']
    else:
        return ['FAKE']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form["message"]
        print(message)
        pred = fake_news_det(message)
        print(pred)
        return render_template('index.html', prediction=pred)
    return None


if __name__ == "__main__":
    app.run(debug=True)
