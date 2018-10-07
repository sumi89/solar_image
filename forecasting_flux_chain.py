#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 00:56:11 2018

@author: sumi
"""

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

flux = np.loadtxt('/Users/sumi/python/research/data/short_long_flux_11_12.txt')    
log_flux = np.log10(flux)

for i in range(1,log_flux.shape[0]):
    if log_flux[i,0]<-9:
        log_flux[i,0]=-9
    if log_flux[i,1]<-7.5:
        log_flux[i,1]=log_flux[i-1,1] 



#
#for i in range(2):
#    f = plt.figure()
#    plt.plot(np.arange(210528), log_flux[:,0], log_flux[:,1])
#    plt.legend(['short_flux', 'long_flux'])
#    plt.xlabel('Time')
#    plt.ylabel('log flux')
#    plt.show()
#    f.savefig("/Users/sumi/Desktop/timestamp.pdf", bbox_inches='tight')

#log_minmax_flux = (log_flux - np.min(log_flux, axis = 0))/(np.max(log_flux, axis = 0) - np.min(log_flux, axis = 0))
long_flux = log_flux[:,1]




min_index = 0
max_index = 210528
#max_index = 20
step = 1
batch_size = 512
#batch_size = 1
lookback = 24
delay = 1
units=16
delay_ = delay*5
#print("lookback = ", lookback,'\t delay={0:4d} minutes'.format(delay_))

#print("lookback = ", lookback, 'future = ', delay_)    
#for lookback in range(3,4):
#    for delay in range(1,25):
#rows = np.arange(min_index + lookback, max_index-4)
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


samples_tr = samples[0:200000, :, :]
targets_tr = targets[0:200000]
samples_val = samples[200000:len(samples), :, :]
targets_val = targets[200000:len(samples)]   

model = models.Sequential()
model.add(LSTM(units=16, activation='tanh', input_shape = (None, samples.shape[-1])))
model.add(Dense(1))
#model.summary()

model.compile(optimizer=RMSprop(lr = 0.0001), loss='mae')
#print("lookback = ", lookback,'\t future={0:4d} minutes'.format(delay_))
history = model.fit(samples_tr, targets_tr, batch_size=batch_size, epochs=20, 
                    validation_data=(samples_val, targets_val),verbose=1)


maes = []
for i in range(1,31):
    print("step=", i, "future = ", i*5)
    pred = model.predict(samples_val)
    error = np.abs(np.reshape(targets_val,(-1,1)) - pred)
    mae = np.mean(error)
    print("mae:", mae)
    maes.append(mae)
    pred_reshape = pred.reshape(len(samples_val),1,1)
    samples_val = np.concatenate((samples_val[:,1:,:],pred_reshape),axis=1)

#maes = maes[39:80]
plt.plot(maes)


#######################################

#pred_list = []
#for i in range(1,12):
#    #print("step=", i)
#    pred = model.predict(samples_val)
#    pred_list.append(pred[0,0])
#    #print("pred", pred)
#    if (i*5)%15==0:
#        print("future=",i*5)
#        print("pred", pred)
#    pred_reshape = pred.reshape(len(samples_val),1,1)
#    samples_val = np.concatenate((samples_val[:,1:,:],pred_reshape),axis=1)
#    #print("samples_val", samples_val)
#
#plt.plot(targets_val[0:11])
#plt.plot(np.array(pred_list))
#plt.legend(['targets_val', 'pred_list'])
#plt.show()
#
#
#error = np.abs(targets_val[0:11] - np.array(pred_list))
#mae = np.mean(error)
#print('MAE', mae)

#######################################


#
#
#loss = history.history['loss']
#val_loss = history.history['val_loss']
#
#epochs = range(len(loss))
#
#plt.figure()
#
#plt.plot(epochs, loss, 'r', label='Training loss')
#plt.plot(epochs, val_loss, 'b', label='Validation loss')
#plt.title('Training and validation loss')
#plt.legend()
#
#plt.show()

######## plotting predicted and actual flux ######
#predicted_flux = model.predict(samples)
#actual_flux = targets
#
#
#plt.figure()
#
#plt.plot(predicted_flux, actual_flux, 'r.', )
#plt.xlabel('Predicted flux', )
#plt.ylabel('actual flux')
#plt.title('Predicted flux and actual flux ')
#plt.legend()
#
#plt.show()      
######## plotting predicted and actual flux (END) #####
#print("loss =", np.array(loss))
#print("val_loss = ", np.array(val_loss))




#x = [0, 1, 2, 3]
#y = [0, 1, 4, 9]
#
#plt.figure()
#plt.plot(x, y, 'r', )
#plt.xlabel('Predicted flux')
#plt.ylabel('actual flux')
#plt.legend()

#plt.show()