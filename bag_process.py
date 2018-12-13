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

class bag_process(object):

	def load(self):
		df = pd.read_csv("data.csv")
		review = []
		output = []
		for index, row in df.iterrows():
			if row[1] == 'Health' or row[1] == 'Opinion':
				review.append(row[0])
				output.append(row[1])
			else:
				review.append(row[0])
				output.append('None')
		return review, output

	def tokenize(self, review, output):
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

	def binarize(self, Y):
		'''
		unique = set(Y)
		print(unique)
		print(len(unique))
		dictionary = {}
		for i, word in enumerate(unique):
			bi = [0] * len(unique)
			bi[i] = 1
			bi=np.flip(np.asarray(bi))
			dictionary.update({word:bi})
		self.Y = []
		for el in Y:
			self.Y.append(dictionary.get(el))
			'''
		self.Y = []
		for y in Y:
			if y == 'Opinion':
				self.Y.append(np.asarray([0,0,1]))
			elif y == 'Health':
				self.Y.append(np.asarray([0,1,0]))
			else:
				self.Y.append(np.asarray([1,0,0]))

	def scale(self):
		self.X = preprocessing.scale(self.X)

	def save(self, bag):
		np.save('input.npy', self.X)
		np.save('output.npy', self.Y)
		with open('bag.pickle', 'wb') as bag_dump:
			pickle.dump(bag, bag_dump)

	def run(self):
		review , output = self.load()
		articles = self.tokenize(review, output)
		bag = self.gen_bag(articles)
		print(bag)
		rows = len(articles)
		colums = len(bag)
		self.X = np.zeros((rows,colums))
		self.convert(articles, bag)
		self.normalize()
		#self.scale()
		self.binarize(output)
		self.save(bag)

gen_bag = bag_process()
gen_bag.run()