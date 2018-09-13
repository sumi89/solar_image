#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 22:13:15 2018

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
import keras

    
### flux #####
flux = np.loadtxt('/Users/sumi/python/research/data/changed_flux_2017/flux_2017.txt')
log_flux = np.log10(flux)
log_minmax_flux = (log_flux - np.min(log_flux, axis = 0))/(np.max(log_flux, axis = 0) - np.min(log_flux, axis = 0))

#### getting images ######
original_dataset_dir = '/Users/sumi/python/research/data/multi_solar_images_trial/'
fnames = [os.path.join(original_dataset_dir, fname) for fname in os.listdir(original_dataset_dir)]

image_height = 472
image_width = 472
channels = 3

image_data = np.ndarray(shape=(8760, image_height, image_width, channels), dtype=np.float32)


i = 0
for img_path in fnames:
    img = image.load_img(img_path, target_size=(472, 472))
    x = image.img_to_array(img).astype(float)
#    x = x/255.
#    x = np.mean(x,axis=2)
#    x = x.reshape(-1)
    image_data[i] = x
    i += 1
    #print(i)
    

vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(472, 472, 3))
vgg16_base.summary()

train_features_no_generator = vgg16_base.predict(image_data)
train_features_no_generator = np.reshape(train_features_no_generator, (8760, 14 * 14 * 512))
np.savetxt('/Users/sumi/python/research/data/features_no_generator_vgg16.txt', train_features_no_generator, delimiter = ',', fmt='%f') 
np.savetxt('/Users/sumi/python/research/data/labels0_no_generator_vgg16.txt', labels0) 

features = np.loadtxt('/Users/sumi/python/research/data/features_no_generator_vgg16.txt', delimiter=",").astype(float)
labels = np.loadtxt('/Users/sumi/python/research/data/labels0_no_generator_vgg16.txt', delimiter=",").astype(float)


train_features = features[0:7000,:]
train_labels = labels0[0:7000]
validation_features = features[7000:8000, :]
validation_labels = labels0[7000:8000]


def generator_lstm(img_data, log_minmax_flux, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), lookback // step, img_data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = img_data[indices]
            #print("each sample", samples[j].shape)
            targets[j] = log_minmax_flux[rows[j] + delay]
        #print("batch", samples.shape)
        yield samples, targets

        
lookback = 3
step = 1
delay = 1
batch_size = 2

val_steps = (1000 - 0 - lookback) // batch_size

train_gen = generator(log_minmax_flux, lookback=lookback, delay=delay, min_index=0, max_index=7000,
                      shuffle=False, step=step, batch_size=batch_size)

val_gen = generator(log_minmax_flux, lookback=lookback, delay=delay, min_index=7001, max_index=8000,
                      shuffle=False, step=step, batch_size=batch_size)


from keras import models
from keras import layers
from keras import optimizers
from keras.optimizers import RMSprop

model = models.Sequential()
model.add(LSTM(units=128, activation='tanh', input_shape = (None, train_features.shape[-1])))
model.add(Dense(1))
model.summary()

# callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=1,),
#                  keras.callbacks.ModelCheckpoint(filepath='my_model.h5', save_best_only=True,)]


model.compile(optimizer=RMSprop(lr = 0.00001), loss='mae')

history = model.fit_generator(train_gen, steps_per_epoch=3500, epochs=20, validation_data=val_gen, validation_steps=val_steps)

import  matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


















