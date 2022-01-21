#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 3 00:14:11 2021

@author: Arina
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import pandas as pd
import random
import glob
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
#%%

#direc = '/Users/Arina/Downloads/FastFile-5kGq3UJYkm6ZaxGN/text_files/'
#direc = '/Users/Arina/Downloads/o1_text_files/'
#direc = '/Users/Arina/Downloads/o1_text_files_2/'
direc = '/Users/Arina/Downloads/o2_text_files/'
txt_files = glob.glob(direc +'Repeating_Blips'+ '*.txt')

tomte1 = [np.loadtxt(x) for x in txt_files]
print("done")
#%%
labels = []

for x in txt_files:
    #file = x.split('.')[0].split('/')[-1].split('_')[0]
    file = x.split('.')[0].split('/')[-1].split('_')
    labels.append(file)
    
#%%

short_labels = []

for x in labels:
    if len(x) == 3:
        x = x[0]
    elif len(x) == 4:
        x = x[0:2]
    elif len(x) == 5:
        x = x[0:3]
    short_labels.append(x)
    
#%%

ylabels = []

for x in short_labels:
    x = ''.join(x)
    ylabels.append(x)
        
#%%    
a = np.array(y)
print(np.unique(a, return_counts=True))
#%%
ds = np.array(ds)
X = ds
X = X.reshape((10485, 128, -1))
#%%

x_train = ds[:812]
x_test = ds[812:]

x_train = x_train.reshape((812, 128, -1))
x_test = x_test.reshape((270, 128, -1))
#%%
#ylabel = np.loadtxt('label_file.txt', dtype=str)
y = y.reshape(-1, 1)

enc = OneHotEncoder(drop=None, sparse=False)
enc.fit(y)
y = enc.transform(y)

'''
y_train = ylabel[:812]
y_train = y_train.reshape(-1, 1)
y_test = ylabel[812:]
y_test = y_test.reshape(-1, 1)

enc = OneHotEncoder(drop='first', sparse=False)
enc.fit(y_train)
enc.fit(y_test)
y_train = enc.transform(y_train)
y_test = enc.transform(y_test)
'''
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=38)
#%%
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

#%%
model = Sequential()
model.add(Conv1D(filters=128, kernel_size=3, activation='relu',input_shape=(128,128)))
#model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(14, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adadelta(
    learning_rate=1.0, rho=0.95, epsilon=1e-07, name="Adadelta"), metrics=['accuracy'])

model.summary()
#%%
# fit network
history = model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=1, validation_data=(X_test, y_test))

#%%
# evaluate model
_, accuracy = model.evaluate(X_test, y_test, batch_size=64, verbose=1)
print(accuracy)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.ylim([0.5, 1])
plt.legend(loc='lower right')

#%%

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#%%
classes = np.array(enc.categories_)
classes = classes.reshape(14)

#%%
y_pred=model.predict(X_test) 
y_test = enc.inverse_transform(y_test)
y_pred = enc.inverse_transform(y_pred)

#y_pred=np.argmax(y_pred, axis=1)
#y_test=np.argmax(y_test, axis=1)
#%%
cm = confusion_matrix(y_test, y_pred, labels = classes, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = classes)
disp.plot()
plt.xticks(fontsize=10, rotation=90)
plt.show()


