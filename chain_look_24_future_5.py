
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
long_flux = long_flux-mn

min_index = 0
max_index = 210528
step = 1
batch_size = 512 #512
lookback = 3
delay = 1
units=16 #16
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

training_size = 150000
samples_tr = samples[0:training_size, :, :]
targets_tr = targets[0:training_size]
samples_val = samples[training_size:len(samples), :, :]
targets_val = targets[training_size:len(samples)]  

model = models.Sequential()
model.add(LSTM(units=units, activation='tanh', input_shape = (None, samples.shape[-1])))
model.add(Dense(1))
model.summary()

lr = 0.01 #0.01
model.compile(optimizer=Adam(lr=lr), loss='mse')
print("batch size = ", batch_size, "units = ", units, "lookback = ", lookback, "learning rate = ", lr, "future = ", delay_, "minutes")

model.fit(samples_tr, targets_tr, batch_size=batch_size, epochs=20, verbose=1)
model.save('/Users/sumi/python/research/models/chain_look24_future5_lr01_b512_u16.h5')

samples_val_chain = samples_val[:-25]
targets_val_chain = targets_val[:-25]


##### check
samples_tr_ = samples[0:10, :, :]
targets_tr_ = targets[0:10]
samples_val_ = samples[10:20, :, :]
targets_val_ = targets[10:20] 

model = models.Sequential()
model.add(LSTM(units=units, activation='tanh', input_shape = (None, samples.shape[-1])))
model.add(Dense(1))
model.summary()

lr = 0.01 #0.01
model.compile(optimizer=Adam(lr=lr), loss='mse')
model.fit(samples_tr_, targets_tr_, batch_size=batch_size, epochs=1, verbose=1)
##### check
samples_val_chain = samples_val_
targets_val_chain = targets_val_


mses = []
for i in range(1,5):
    print("step=", i, "future = ", i*5)
    #print('samples_val_chain:', samples_val_chain)
    #print('samples_val_chain shape :', samples_val_chain.shape)
    #print("targets_val_chain:", targets_val_chain)
    #print('targets_val_chain shape :', targets_val_chain.shape)
    pred = model.predict(samples_val_chain)
    #print('pred', pred)
    target_val_reshape = np.reshape(targets_val_chain,(-1,1))
    error = np.abs(target_val_reshape - pred)
    #mae = np.mean(error)
    #print("mae:", mae) 
    #maes.append(mae)
    squared_error = error * error
    mse = np.mean(squared_error)
    print("mse:", mse)
    mses.append(mse)
    pred_reshape = pred.reshape(len(samples_val_chain),1,1)
    # appending the prection in the samples_val_chain
    samples_val_chain = np.concatenate((samples_val_chain, pred_reshape),axis=1)
    # appending the next flux in the tagrget_val chain
    #targets_val_chain = np.append(target_val_reshape, targets[i-1+len(samples_val_chain)])

#plt.plot(mses)





#mses = []
#for i in range(1,6):
#    print("step=", i, "future = ", i*5)
#    print('samples_val_chain:', samples_val_chain)
#    print('samples_val_chain shape :', samples_val_chain.shape)
#    targets_val_chain = targets_val[i-1:i-1+len(samples_val_chain)]
#    print("targets_val_chain:", targets_val_chain)
#    print('targets_val_chain shape :', targets_val_chain.shape)
#    pred = model.predict(samples_val_chain)
#    print('pred', pred)
#    target_val_reshape = np.reshape(targets_val_chain,(-1,1))
#    error = np.abs(target_val_reshape - pred)
#    #mae = np.mean(error)
#    #print("mae:", mae) 
#    #maes.append(mae)
#    squared_error = error * error
#    mse = np.mean(squared_error)
#    print("mse:", mse)
#    mses.append(mse)
#    pred_reshape = pred.reshape(len(samples_val_chain),1,1)
#    # appending the prection in the samples_val_chain
#    samples_val_chain = np.concatenate((samples_val_chain[:,1:,:],pred_reshape),axis=1)
#    # appending the next flux in the tagrget_val chain
#    #targets_val_chain = np.append(target_val_reshape, targets_val[i-1+len(samples_val_chain)])
#
##plt.plot(mses)
