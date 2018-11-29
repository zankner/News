import pandas as pd
import numpy as np
import matplotlib as plt
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
import pickle
import sys

class pre_process(object):

    def __init__(self, train_length, input, output):
        self.X = []
        self.Y = []
        self.input = input
        self.output = output
        self.train_length = train_length
        self.bag = []

    def load(self):
        df = pd.read_csv("data.csv")
        review = df.iloc[:,0]
        output =  df.iloc[:,1]
        return review, output

    def tokenize(self, review, output):
        articles = []
        for input in review:
            #tokenize
            t = nltk.word_tokenize(str(input))
            articles.append(t)
        return articles

    def gen_bag(self, articles):
        bag = []
        stemmer = PorterStemmer()
        for sentence in articles:
            for word in sentence:
                pattern_word = stemmer.stem(word.lower())
                bag.append(pattern_word)
        bag = list(set(bag))

        badwords = set(stopwords.words('english'))
        bag = [b for b in bag if not b in badwords]
        return bag


    def series(self, bag, articles, review, output):
        X = []
        Y = []
        for a,y in zip(articles, output):
            for word in range(0, len(a) - self.train_length):
                input = a[word : (word + self.train_length)]
                output = y
                X.append(input)
                Y.append(str(y))

        for x,y in zip(X,Y):
            to_add = []
            for word in x:
                if(word.lower() in bag):
                    to_add.append(1)
                else:
                    to_add.append(0)
            self.X.append(to_add)
            self.Y.append(y)


    def pad(self):
        self.X = pad_sequences(np.asarray(self.X), padding='post')
        lb = preprocessing.LabelBinarizer()
        self.Y = lb.fit_transform(np.asarray(self.Y))


    def save(self):
        np.save(self.input, self.X)
        np.save(self.output, self.Y)

    def process(self):
        review , output = self.load()
        articles = self.tokenize(review, output)
        bag = self.gen_bag(articles)
        self.series(bag, articles, review, output)
        self.pad()
        self.save()
