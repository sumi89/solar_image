#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 22:52:22 2018

@author: sumi
"""


from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras.optimizers import RMSprop, Adam
import  matplotlib.pyplot as plt
from keras.layers.wrappers import TimeDistributed
from keras.layers import Conv2D, MaxPooling2D, LSTM, Convolution2D, Dense

flux = np.loadtxt('/Users/sumi/python/research/data/short_long_flux_11_12.txt')    
log_flux = np.log10(flux)

for i in range(1,log_flux.shape[0]):
    if log_flux[i,0]<-9:
        log_flux[i,0]=-9
    if log_flux[i,1]<-7.5:
        log_flux[i,1]=log_flux[i-1,1] 

long_flux = log_flux[:,1]
mn = np.mean(long_flux[:150000])
sd = np.std(long_flux[:150000])

long_flux = long_flux-mn

min_index = 0
max_index = 210528
#max_index = 20
step = 1
batch_size = 512
#batch_size = 1
lookback = 24
delay = 12
units=256
kernel=23
delay_ = delay*5

rows = np.arange(min_index + lookback, max_index-delay+1)
#print('rows', rows)

samples = np.zeros((len(rows), lookback // step,))
targets = np.zeros((len(rows),))
#for delay in range(1,25):
for j, row in enumerate(rows):
#    print('j', j)
#    print('row', row)
#    print('rows[j]', rows[j])
    indices = range(rows[j] - lookback, rows[j], step)
#    print('indices', indices)
    samples[j] = long_flux[indices]
#    print("samples",samples[j])
    #for delay in range(1,25):
    targets[j] = long_flux[rows[j] + delay-1]
#    print("targets", targets[j])
#    print("=======")
samples = samples.reshape(samples.shape[0],samples.shape[1],1)
#print("#########")
#print('samples shape', samples.shape)
#print('target shape', targets.shape)

training_size = 150000
samples_tr = samples[0:training_size, :, :]
targets_tr = targets[0:training_size]
samples_val = samples[training_size:len(samples), :, :]
targets_val = targets[training_size:len(samples)]   

samples_val = samples_val[:-25]
targets_val = targets_val[:-25]

model = models.Sequential()
model.add(layers.Conv1D(units, kernel, activation = 'relu', input_shape = (None, samples.shape[-1])))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
model.summary()

lr = 0.01
model.compile(optimizer=Adam(lr = lr), loss='mse')
print("batch size = ", batch_size, "units = ", units, "kernel = ", kernel, "lookback = ", lookback, "learning rate = ", lr, "future = ", delay_)

history = model.fit(samples_tr, targets_tr, batch_size=batch_size, epochs=20, 
                    validation_data=(samples_val, targets_val),verbose=1)

model.save('/Users/sumi/python/research/models/conv1d_look24_future5_lr01_b512_u256_k23.h5')





