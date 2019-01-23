#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 00:45:33 2019

@author: sumi
"""

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dense, Dropout, Flatten
from keras import backend as K
from keras.optimizers import RMSprop, Adam
from keras.initializers import he_normal
import numpy as np
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import pandas as pd

#import sys
#sys.path.append('/Users/sumi/anaconda3/pkgs/keras-preprocessing-1.0.2-py_1/site-packages/keras_preprocessing/')

input_shape = (512,512,1)
input_shape = (256,256,1)
#main_dir = "None"
target_name =['P>1', 'P>5', 'P>10', 'P>30', 'P>50', 'P>100']

#dataframe_dir = '/Volumes/New Volume/code/'
#pdf = pd.read_excel('/Volumes/New Volume/code/images_and_target_log.xlsx')
#pdf['filename'] = pdf['filename'].str.replace('/media/ofuentes/','/Volumes/')
#pdf.to_excel('/Users/sumi/python/research/research_proton_flux/images_and_target_log.xlsx')
pdf = pd.read_excel('/Users/sumi/python/research/research_proton_flux/images_and_target_log.xlsx')
pdf['filename'] = pdf['filename'].str.replace('/Volumes/New Volume/','/Volumes/rapid/')
#pdf1 = pdf[0:100]

#pdf1 = pd.read_excel('/Users/sumi/Documents/trial.xlsx')
datagen=ImageDataGenerator(rescale=1./255)
train_generator=datagen.flow_from_dataframe(dataframe=pdf, directory = "None", color_mode= "grayscale" , 
                                            x_col="filename", y_col="P>100", class_mode="other", 
                                            target_size=input_shape[:2], batch_size=32)

model = Sequential()
model.add(Conv2D(50, kernel_size=(3, 3), kernel_initializer = he_normal(), input_shape=input_shape))
model.add(BatchNormalization())
#model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(75, (3, 3), kernel_initializer = he_normal(),activation='relu'))
model.add(BatchNormalization())
#model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(110, (3, 3), kernel_initializer = he_normal(),activation='relu'))
model.add(BatchNormalization())
#model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(160, (3, 3), kernel_initializer = he_normal(),activation='relu'))
model.add(BatchNormalization())
#model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2),padding="same"))

model.add(Conv2D(240, (3, 3), kernel_initializer = he_normal(),activation='relu'))
model.add(BatchNormalization())
#model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2),padding="same"))

model.add(Conv2D(32, (1, 1), kernel_initializer = he_normal(),activation='relu'))
model.add(Flatten())
#model.add(Dropout(0.2))
model.add(Dense(64,kernel_initializer = he_normal(), activation='relu'))
model.add(Dense(1, kernel_initializer = he_normal(),activation='linear'))

model.summary()

model.compile(loss='mse',
              optimizer=keras.optimizers.Adam())

#STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_TRAIN = 100
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=10)






















