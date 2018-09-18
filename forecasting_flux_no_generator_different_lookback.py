#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 01:28:36 2018

@author: sumi
"""

import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras.optimizers import RMSprop
import  matplotlib.pyplot as plt
from keras.layers.wrappers import TimeDistributed
from keras.layers import Conv2D, MaxPooling2D, LSTM, Convolution2D, Dense

flux = np.loadtxt('/Users/sumi/python/research/data/changed_flux_2016/flux_2016.txt')
log_flux = np.log10(flux)
log_minmax_flux = (log_flux - np.min(log_flux, axis = 0))/(np.max(log_flux, axis = 0) - np.min(log_flux, axis = 0))
labels0 = log_minmax_flux[:,0]



#for future in range(1, 25):
#    baseline_prev = np.mean(np.abs(log_minmax_flux[future:, 0]-log_minmax_flux[:-future, 0]))
#    print("future = ", future, "MAE", baseline_prev)
#
#for future in range(1, 25):
#    baseline_prev = np.mean(np.abs(labels0[future:]-labels0[:-future]))
#    print("future = ", future, "MAE", baseline_prev)



min_index = 0
max_index = 8000
step = 1
batch_size = 1
lookback = 24
delay = 1

print("lookback = ", lookback, 'future = ', delay)    
#for lookback in range(3,4):
#    for delay in range(1,25):
rows = np.arange(min_index + lookback, max_index-1)
#print('rows', rows)

samples = np.zeros((len(rows),
                   lookback // step,
                   log_minmax_flux.shape[-1]))
targets = np.zeros((len(rows),))
#for delay in range(1,25):
for j, row in enumerate(rows):
#    print('j', j)
#    print('row', row)
#    print('rows[j]', rows[j])
    indices = range(rows[j] - lookback, rows[j], step)
#    print('indices', indices)
    samples[j] = log_minmax_flux[indices]
#    print("samples",samples[j])
    #for delay in range(1,25):
    targets[j] = labels0[rows[j] + delay]
#    print("targets", targets[j])
#    print("=======")
#print("#########")


samples_tr = samples[0:7000, :, :]
targets_tr = targets[0:7000]
samples_val = samples[7000:len(samples), :, :]
targets_val = targets[7000:len(samples)]   

model = models.Sequential()
model.add(LSTM(units=4, activation='tanh', input_shape = (None, samples.shape[-1])))
model.add(Dense(1))
model.summary()

model.compile(optimizer=RMSprop(lr = 0.0001), loss='mae')

print("lookback = ", lookback, 'future = ', delay) 
history = model.fit(samples_tr, targets_tr,
          batch_size=batch_size,
          epochs=20,
          validation_data=(samples_val, targets_val),
          verbose=1)



loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

######## plotting predicted and actual flux ######
predicted_flux = model.predict(samples)
actual_flux = targets
    
plt.figure()

plt.plot(predicted_flux, actual_flux, 'r.', )
plt.xlabel('Predicted flux', )
plt.ylabel('actual flux')
#plt.title('Predicted flux and actual flux ')
plt.legend()

plt.show()      
######## plotting predicted and actual flux (END) #####
print("loss =", np.array(loss))
print("val_loss = ", np.array(val_loss))




#x = [0, 1, 2, 3]
#y = [0, 1, 4, 9]
#
#plt.figure()
#plt.plot(x, y, 'r', )
#plt.xlabel('Predicted flux')
#plt.ylabel('actual flux')
#plt.legend()

#plt.show()