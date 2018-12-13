import pandas as pd
import numpy as np
import matplotlib as plt
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
from keras.utils import to_categorical
import pickle
import sys

class flask_process(object):

	def __init__(self, csv, text):
		self.csv = csv
		self.text = text
		with open('/Users/zackankner/Desktop/Github/News/bag.pickle', 'rb') as input_file:
			self.dictionary = pickle.load(input_file)
		#self.dictionary = np.load('/Users/zackankner/Desktop/Github/News/bag.npy')

	def load(self):
		if self.csv == True:
			df = pd.read_csv("data.csv")
			review = df.iloc[:,0]
			output =  df.iloc[:,1]
			return review, output
		else:
			review = []
			review.append(self.text)
			return review

	def tokenize(self, review):
		articles = []
		for input in review:
			#tokenize
			t = nltk.word_tokenize(str(input))
			articles.append(t)
		return articles

	def gen_bag(self, text):
		stemmer = PorterStemmer()
		badwords = set(stopwords.words('english'))
		bag = {}
		counter = 0
		for article in text:
			for word in article:
				stemmed = stemmer.stem(word.lower())

				if stemmed not in bag:
					if stemmed not in badwords:
						bag.update({stemmed:counter})
						counter += 1
		return bag

	def convert(self, text, dictionary):
		stemmer = PorterStemmer()
		for i, article in enumerate(text):

			for word in article:
				if stemmer.stem(word.lower()) in dictionary:
					index = dictionary.get(stemmer.stem(word.lower()))
					self.X[i][index] += 1

	def normalize(self):
		X = np.log(self.X)
		for i, row in enumerate(X):
			for j, element in enumerate(row):
				if element <= 0:
					X[i][j] = 0
		self.X = X


	def scale(self):
		self.X = preprocessing.scale(self.X)

	def save(self):
		np.save('record.npy', self.X)
		if self.csv:
			np.save('output.npy', self.Y)

	def run(self):
		if self.csv == True:
			review , output = self.load()
		else:
			review = self.load()
		articles = self.tokenize(review)
		rows = len(articles)
		colums = len(self.dictionary)
		self.X = np.zeros((rows,colums))
		self.convert(articles, self.dictionary)
		self.normalize()
		self.save()

#gen_bag = bag_process(False, "Figh Me Looser Kid")
#gen_bag.run()