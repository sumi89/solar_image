#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 22:04:56 2018

@author: sumi
"""

import os, shutil

from keras import layers
from keras import models
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop


### flux #####
flux = np.loadtxt('/Users/sumi/python/research/data/changed_flux_2017/flux_2017.txt')
log_flux = np.log10(flux)
log_minmax_flux = (log_flux - np.min(log_flux, axis = 0))/(np.max(log_flux, axis = 0) - np.min(log_flux, axis = 0))

def generator(data, lookback, delay, min_index, max_index,
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

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][0]
        yield samples, targets
        
lookback = 3
step = 1
delay = 1
batch_size = 2

train_gen = generator(log_minmax_flux, lookback=lookback, delay=delay, min_index=0, max_index=7000,
                      shuffle=False, step=step, batch_size=batch_size)

val_gen = generator(log_minmax_flux, lookback=lookback, delay=delay, min_index=7001, max_index=8000,
                      shuffle=False, step=step, batch_size=batch_size)

val_steps = (8000 - 7001 - lookback) // batch_size

## baseline with the average ##
def evaluate_naive_method_avg():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = np.mean(samples[:, :, 0], axis = 1)
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print("Baseline result using average ", np.mean(batch_maes))
    
evaluate_naive_method_avg()
## baseline with the average(end) ##

## baseline with the previous one hour ##
baseline_prev = np.mean(np.abs(log_minmax_flux[1:,0]-log_minmax_flux[:-1,0]))
print("Baseline result using previous one flux ", baseline_prev)
## baseline with the previous one hour ##

## forecasting flux with the LSTM ##
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
## forecasting flux with the LSTM(end) ##
