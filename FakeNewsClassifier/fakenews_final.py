#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 00:06:13 2018

@author: vivek
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.externals import joblib
import re

df = pd.read_csv("~/Downloads/train.csv")
df = df.dropna()
data = df["title"]
data = data.apply(lambda x: x.lower())

batch_size = 5000
tokenizer = Tokenizer(nb_words=batch_size, split=' ')

tokenizer.fit_on_texts(data.values)
X = tokenizer.texts_to_sequences(data.values)

maxx = 0
c=0
for ll in X:
    c=len(ll)
    if c>maxx:
        maxx=c
print('Max',maxx)
X = pad_sequences(X)
print(len(X[0]))

embed_dim = 128
lstm_out = 256
model = Sequential()
model.add(Embedding(batch_size, embed_dim,input_length = X.shape[1], dropout=0.2))
model.add(LSTM(lstm_out, dropout_U=0.2, dropout_W=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


Y = pd.get_dummies(df['label']).values
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

model.fit(X_train, Y_train, nb_epoch = 2, batch_size=64, verbose = 2)

def predict_news(inp):
    input_val = inp
    z = tokenizer.texts_to_sequences([input_val])
    z = pad_sequences(z, maxlen=59)
    z = np.array(z[0]).reshape(1, 59)
    pred = model.predict(z)
    if pred[0][0] > 0.5:
        result="true"
    else:
        result="fake"
    return result

print(predict_news("Pope has a new baby"))
print(predict_news("Pope is praying"))

model.save('fake_news.h5')