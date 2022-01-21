#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 00:36:37 2022

@author: Arina
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Cropping2D
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
#%% Load data

ds = np.load('brscaleograms.npy')

#ds = np.reshape(ds, (1082,480,640, 1))

#%% Preprocessing

ds = ds / 255.0

#%% Load labels

y = np.load('downsbrlabels.npy')

#%% Show first picture from each class

classes = np.array(['AirCompressor', 'Blip', 'ExtremelyLoud', 'KoiFish',
       'LowFrequencyBurst', 'LowFrequencyLines', 'PowerLine',
       'RepeatingBlips', 'ScatteredLight', 'Scratchy', 'Tomte', 'Whistle'])

for i in classes:
    examples = np.where(y == i)
    first_index = examples[0][0]
    plt.imshow(ds[first_index])
    plt.xlabel(i)
    plt.figure()

    

#%% Encode labels

y = y.reshape(-1, 1)

enc = OneHotEncoder(drop=None, sparse=False)
enc.fit(y)
y = enc.transform(y)

#%% Separate data

X_train, X_test, y_train, y_test = train_test_split(ds, y, test_size=0.20, random_state=38)

#%%
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

#%%

model = Sequential()
model.add(Cropping2D(cropping=((20, 20), (20, 100)), input_shape=(480, 640, 3)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding="same" ))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(2))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

#%%

history = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1, validation_data=(X_test, y_test))

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











