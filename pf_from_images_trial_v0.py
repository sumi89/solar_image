#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 00:25:20 2019

@author: sumi
"""

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

input_shape = (8,8,3)

#main_dir = "None"
target_name =['P>1', 'P>5']

#dataframe_dir = '/Volumes/New Volume/code/'
#pdf = pd.read_excel('/Volumes/New Volume/code/images_and_target_log.xlsx')
#pdf['filename'] = pdf['filename'].str.replace('/media/ofuentes/','/Volumes/')
#pdf.to_excel('/Users/sumi/python/research/research_proton_flux/images_and_target_log.xlsx')
#pdf = pd.read_excel('/Users/sumi/python/research/research_proton_flux/images_and_target_log.xlsx')
#pdf['filename'] = pdf['filename'].str.replace('/Volumes/New Volume/','/Volumes/rapid/')
#pdf1 = pdf[0:100]

pdf1 = pd.read_excel('/Users/sumi/Documents/trial.xlsx')
pdf1_train = pdf1[0:10]
pdf1_valid = pdf1[10:14]
pdf1_test = pdf1_valid.drop(["P>1","P>5"], axis = 1)

def ch_making_func(im):
    #img = image.load_img(im, target_size = (8,8))
    im_arr = image.img_to_array(im).astype(float)
    im_arr[:,:,0] = im_arr[:,:,0]
    im_arr[:,:,1] = np.mean(im_arr,axis=2)
    im_arr[:,:,2] = np.var(im_arr,axis=2)
    return im_arr

datagen=ImageDataGenerator(rescale=1./255, preprocessing_function = ch_making_func)
train_generator=datagen.flow_from_dataframe(dataframe=pdf1_train, directory = "None", color_mode= "rgb" , 
                                            x_col="filename", y_col=target_name, class_mode="other", 
                                            target_size=input_shape[:2], batch_size=2)

valid_generator=datagen.flow_from_dataframe(dataframe=pdf1_valid, directory = "None", color_mode= "rgb" , 
                                            x_col="filename", y_col=target_name, class_mode="other", 
                                            target_size=input_shape[:2], batch_size=2)

test_generator=datagen.flow_from_dataframe(dataframe=pdf1_test, directory = "None", color_mode= "rgb" , 
                                            x_col="filename", y_col="None", class_mode=None, 
                                            target_size=input_shape[:2], batch_size=2)

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
model.add(Dense(2, kernel_initializer = he_normal(),activation='linear'))

model.summary()

model.compile(loss='mse',
              optimizer=keras.optimizers.Adam())

#STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_TRAIN = 2
STEP_SIZE_VALID = 2
history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=5)

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


## Predict the output
test_generator.reset()
pred=model.predict_generator(test_generator,verbose=1)




















