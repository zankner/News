from flask import Flask, request, render_template
import numpy as np 
import keras.models
import tensorflow as tf
import os
import re
import sys
from scipy.misc import imsave, imread, imresize
sys.path.append(os.path.abspath('../'))
from keras.models import load_model
sys.path.insert(0, '/Users/zackankner/Desktop/Github/News')
import model as md
import flask_process as fp
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard



#initalize flask and static foldeer
app = Flask(__name__, static_folder='static')


APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("mainpage.html")


@app.route('/uploadtext', methods=['GET', 'POST'])
def uploadtext():
    return render_template("uploadtext.html")

def predict(article):
    model = md.model()
    flask_process = fp.flask_process(False, article)
    flask_process.run()
    prediction, values = model.report(text = np.load('./record.npy'))
    return prediction, values

@app.route('/classify', methods = ['GET', 'POST'])
def classify():

    text = request.form['articletext']

    #INSERT CLASSIFIER HERE
    #have the classifier spit out the accuracy in percent and the classified article category
    
    guess, complete = predict(text)

    if guess == 2:
        category = 'Opinion'
    elif guess == 1:
        category = 'Health'
    else:
        category = 'Neither'

    acc = int(complete.item(guess) * 100)

    print(acc)

    return render_template("resultpage.html", accuracy = acc, article_catg = category, og_text = text)







if __name__ == '__main__':

    app.run(debug=True, port= 5001)
