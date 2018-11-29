import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import numpy as np
import pandas as pd


class model(object):
    def __init__(self, train_length,lstm_size,dropout):
        self.X = np.load('input.npy')
        self.Y = np.load('output.npy')
        print(self.Y.shape)
    #    df = pd.read_csv('data.csv')
    #    self.Y = df.iloc[:,1]
        self.train_length = train_length
        self.lstm_size = lstm_size
        self.dropout = dropout

    def convert_output(self):
        counter = 0
        dict = {}
        for y in set(self.Y):
            dict.update({y:counter})
            counter += 1
        counter = 0
        for y in self.Y:
            self.Y[counter] = dict.get(y)
            counter += 1

        print(self.Y)


    def model_arhitecture(self):
        len = np.size(self.X,0)
        X = np.reshape(self.X, (len, self.train_length, 1))
        print('----------- Running Lstm ------------')
        model = Sequential()
        model.add(LSTM(self.lstm_size, return_sequences = True, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dropout(self.dropout))
        model.add(LSTM(self.lstm_size))
        model.add(Dropout(self.dropout))

        model.add(Dense(self.Y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        tensorboard = TensorBoard(log_dir="logs/{}", histogram_freq=0, batch_size=32, write_graph=False, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

        base = "models/"
        filepath = base + "model.hdf5"

        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='min')
        model.fit(X, self.Y, epochs=1000, batch_size=128, callbacks=[checkpoint, tensorboard])

model = model(20, 1, 0.2)
#smodel.convert_output()
model.model_arhitecture()
