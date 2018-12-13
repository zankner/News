import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


class model(object):
    def __init__(self):
        self.X = np.load('/Users/zackankner/Desktop/Github/News/input.npy')
        self.Y = np.load('/Users/zackankner/Desktop/Github/News/output.npy')
        self.batch_size = 300
        self.epochs = 200
        self.train_X, self.val_X, self.train_Y, self.val_Y = train_test_split(self.X, self.Y, test_size = .5)
        print(self.train_X.shape, 'print')
        #self.X = np.random.rand()
        #self.train_X = self.X[:75]
        #self.val_X = self.X[:-225]
        #self.train_Y = self.Y[:75]
        #self.val_Y = self.Y[:-225]

    def run(self):
        model = Sequential()

        model.add(Dense(100, input_dim=17469, activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(Dense(50, activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(20, activation = 'relu'))
        model.add(Dense(3, activation = 'sigmoid'))
        model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics = ['accuracy'])
        model.summary()

        tensorboard = TensorBoard(log_dir="logs/{}", histogram_freq=0, batch_size=self.batch_size, write_graph=False, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
        filepath = 'checkpoints/model.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=2, save_best_only=True, mode='min')
        model.fit(self.train_X, self.train_Y, epochs=self.epochs, batch_size=self.batch_size, callbacks=[checkpoint, tensorboard], validation_data=(self.val_X, self.val_Y))

    def report(self, text):
        model = Sequential()

        model.add(Dense(100, input_dim=17469, activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(Dense(50, activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(20, activation = 'relu'))
        model.add(Dense(3, activation = 'sigmoid'))

        model.load_weights('../checkpoints/model.h5')

        model.compile(loss='mean_squared_error', optimizer='RMSprop')

        prediction = model.predict(text)
        print(np.argmax(prediction))
        return np.argmax(prediction), prediction
'''
when training un copy
model = model()
model.run()
'''
