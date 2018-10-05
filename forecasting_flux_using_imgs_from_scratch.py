#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 13:24:49 2018

@author: sumi
"""

import os, shutil

from keras import layers
from keras import models
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, LSTM
from keras.layers import Dense, Dropout, Flatten
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

import matplotlib.pyplot as plt
import numpy as np
import pickle

import glob
import requests
import re
import urllib.request
from urllib.request  import urlopen
import cv2 
from PIL import Image
import requests
import io
from io import BytesIO
from urllib.parse import urlparse
from bs4 import BeautifulSoup, SoupStrainer
import datetime
from itertools import chain
import math

from keras import applications
from keras.applications import VGG16, InceptionV3
from keras.applications.resnet50 import ResNet50
import keras

from keras.layers import Conv2D, MaxPooling2D, LSTM, Convolution2D
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras import optimizers

    
### flux #####
flux = np.loadtxt('/Users/sumi/python/research/data/short_long_flux_11_12.txt')    
log_flux = np.log10(flux)

for i in range(1,log_flux.shape[0]):
    if log_flux[i,0]<-9:
        log_flux[i,0]=-9
    if log_flux[i,1]<-7.5:
        log_flux[i,1]=log_flux[i-1,1] 

long_flux = log_flux[:,1]
long_flux_2012 = long_flux[105120:]
long_flux_2012_processed = np.zeros(8784)
#aaaa_ = np.array([0,1,2,3,4,5,6,7,8,9,10,11])
for i in range(len(long_flux_2012_processed)):
    #print(i)
    start = i*12
    end = (i+1)*12
    #print(aaaa_[start:end])
    long_flux_2012_processed[i] = np.mean(long_flux_2012[start:end])
    #print(aaaa_[i])


#### getting original images ######
original_dataset_dir = '/Users/sumi/python/research/data/solar_images_2012_multiwavelength/'
fnames = [os.path.join(original_dataset_dir, fname) for fname in os.listdir(original_dataset_dir)]

image_height_sized = 236
image_width_sized = 236
channels = 3

image_data_resized = np.ndarray(shape=(8784, image_height_sized, image_width_sized, channels), dtype=np.float32)

i = 0
for img_path in fnames:
    img = image.load_img(img_path, target_size=(236, 236))
    x = image.img_to_array(img).astype(float)
#    x = x/255.
#    x = np.mean(x,axis=2)
#    x = x.reshape(-1)
    image_data_resized[i] = x
    i += 1
    print(i)

train_features = image_data_resized[0:7000, :, :, :]
train_labels = long_flux_2012_processed[0:7000]
validation_features = image_data_resized[7000:8000, :, :, :]
validation_labels = long_flux_2012_processed[7000:8000]

#Build model  
model = Sequential()
model.add(Conv2D(64, (5, 5), activation='relu', input_shape= (236, 236, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (5, 5), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (5, 5), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (5, 5), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1))
model.summary()

model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='mae')
history = model.fit(train_features, train_labels, epochs=10, batch_size=20, validation_data=(validation_features, validation_labels))
