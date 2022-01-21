#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 19:30:45 2021

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
import pywt
import scaleogram as scg
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import imageio
from moviepy.video.io.bindings import mplfig_to_npimage
from scipy.interpolate import interp1d

#%%
plt.ioff()
#%%

direc = '/Users/Arina/Downloads/FastFile-5kGq3UJYkm6ZaxGN/text_files/'
#direc = '/Users/Arina/Downloads/o1_text_files/'
#direc = '/Users/Arina/Downloads/o1_text_files_2/'
#direc = '/Users/Arina/Downloads/o2_text_files/'

txt_files = glob.glob(direc + '*.txt')

ds = [np.loadtxt(x) for x in txt_files]
print("done")
\#%%

#ds = np.array(ds)
#time = np.linspace(0, 4, num=16384, endpoint=False)
time = np.linspace(0, 4, num=4096, endpoint=False)
#%%
scales = np.arange(1, 256)

#%%

def downsample(array, npts):
    interpolated = interp1d(np.arange(len(array)), array, axis = 0, fill_value = 'extrapolate')
    downsampled = interpolated(np.linspace(0, len(array), npts))
    return downsampled
#%%
downs = np.ndarray(shape=(1082, 4096))

for ii in range(0,546):
    signal = ds[ii, :]
    downsampled_X = downsample(signal, 4096)
    downs[ii, :] = downsampled_X 
#%%
#im = imageio.imread('exampleglitch.png')
#print(im.shape)

#%%
def genimg(x, ds, time, scales, wav): #x in an integer - index of a particular signal in dataset

    fig = plt.figure()
    ax = fig.add_subplot(111)
    scg.cws(time, wavelet=wav, signal=ds[x], scales=scales, yaxis='frequency', xlim=(1.5, 2.5), ylim=(12,300), yscale='log', spectrum='amp', ax=ax)
    #ax.plot()
    #plt.axis('off')
    #plt.tight_layout()
    #plt.margins(x=0, y=0)
    #plt.close()

    
    numpy_fig = mplfig_to_npimage(fig)
    return numpy_fig
    
#%%
    
#X = []

for x in range(500, 546):
    
    numpy_fig = genimg(x, downs, time, scales, 'cgau1')
    X.append(numpy_fig)
    

#%%

#X = []
for x in line_indx:
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scg.cws(time, wavelet='cgau1', signal=ds[x], scales=scales, yaxis='frequency', xlim=(1.8, 2.2), ylim=(10,1000), spectrum='power', ax=ax)

    plt.axis('off')
    plt.tight_layout()
    plt.margins(x=0, y=0)
    plt.close()
    plt.clf()
    
    numpy_fig = mplfig_to_npimage(fig)
    numpy_fig = numpy_fig[:,:,2]
    X.append(numpy_fig)

#%%
X = np.array(X)

with open('imagedata-o1-downsample.npy', 'ab') as f:
    np.save(f, X)
#%%

loaded_array = np.load('imagedata-o1-downsample.npy')

#%%

np.savetxt('downsbrlabels.txt', brlabels, fmt='%s')







