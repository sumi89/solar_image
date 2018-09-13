#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 22:12:27 2018

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



def get_urls_of_imgs_w_wavelength_resolution(date1, date2, wavelength, resolution, url):
    required_urls = []
    start_date = datetime.datetime.strptime(date1, '%Y-%m-%d %X')
    #start_date = start_date.strftime('%Y/%m/%d')
    end_date = datetime.datetime.strptime(date2, '%Y-%m-%d %X')
    step = datetime.timedelta(days = 1)
    while start_date <= end_date:
        #date = start_date.date()
        ##print (start_date.date())
        dt = start_date.date()
        dt_f = dt.strftime('%Y/%m/%d')
        url_d = url + dt_f + '/'
        #url_dates.append(url_d)
        #start_date += step
    
    # this method will take the url with date, return the url with date and image file name (with wavelength)
    #urls_dates_images = []
    #for i in range(len(url_dates)):
    #    page = requests.get(url_dates[i])    
        page = requests.get(url_d) 
        data = page.text
        soup = BeautifulSoup(data)
        # get the image file name
        img_files=[]    # image files with all info like wavelength, resolution, time
        for link in soup.find_all('a'):
            img_file = link.get('href')
            img_files.append(img_file)
            
        img_files_wr = []        # image files with all info like wavelength, resolution   
        for k in range(5, len(img_files)):
            splitting = re.split(r'[_.?=;/]+',img_files[k])
            if (splitting[3] == wavelength and splitting[2] == resolution):
                img_files_wr.append(img_files[k])
        
        hrs_url = np.zeros(len(img_files_wr))
        for time in range(len(img_files_wr)):
            url_split = re.split(r'[_.?=;/]+',img_files_wr[time])
            hr_min_sec = url_split[1]
            hrs_url[time] = float(hr_min_sec[0:2]) + float(hr_min_sec[2:4])/60 + float(hr_min_sec[4:6])/3600       
            
        start_hr = 0
        index_hrs_url = 0
        for hours in range(0, 24):
            #print("start_hr", hours)
            diff = abs(hrs_url[start_hr:len(img_files_wr)] - hours)
            if len(diff) != 0:
                #print("diff", diff)
                index = np.argmin(diff)
                #print('index', index)
                if (index == 0):
                    index_hrs_url += index 
                    #print("index_hrs_url",index_hrs_url)
                else:
                    index_hrs_url += index + 1
                    #print("index_hrs_url", index_hrs_url)
            else:
                index = index_hrs_url
                #print("index_hrs_url", index_hrs_url)
                #print("index", index)
            start_hr = index_hrs_url + 1
            #required_urls.append(img_files_wr[index_hrs_url])  
            url_date_wave_res = url_d + img_files_wr[index_hrs_url]
            required_urls.append(url_date_wave_res) 
            
        start_date += step  
    return required_urls

date1 = '2017-01-01 00:00:00'
date2 = '2017-12-31 23:59:59'
url = "https://sdo.gsfc.nasa.gov/assets/img/browse/"
wavelength1 = '0131'
wavelength2 = '1600'
wavelength3 = 'HMII'
resolution = '512'

required_urls1 = get_urls_of_imgs_w_wavelength_resolution(date1, date2, wavelength1, resolution, url)
required_urls2 = get_urls_of_imgs_w_wavelength_resolution(date1, date2, wavelength2, resolution, url)
required_urls3 = get_urls_of_imgs_w_wavelength_resolution(date1, date2, wavelength3, resolution, url)

image_height = 472
image_width = 472
channels = 3
image_data = np.ndarray(shape=(len(required_urls1), image_height, image_width, channels), dtype=np.float32)

#image_data = np.ndarray(shape=(5, image_height, image_width, channels), dtype=np.float32)
# this method will take the url with date and image name, return the corresponding images 
#img_all=[]
#j = 0
for i in range(0,len(required_urls1)):
#for i in range(0,5):
    print('i=',i)
    response1 = requests.get(required_urls1[i])
    response2 = requests.get(required_urls2[i])
    response3 = requests.get(required_urls3[i])
    img1 = Image.open(BytesIO(response1.content))
    img2 = Image.open(BytesIO(response2.content))
    img3 = Image.open(BytesIO(response3.content))
    img1 = np.array(img1) # img.shape: height x width x channel
    img1 = img1/255        # scaling from [0,1]
    img1 = np.max(img1,axis=2) #take the mean of the R, G and B  
    img2 = np.array(img2) # img.shape: height x width x channel
    img2 = img2/255        # scaling from [0,1]
    img2 = np.max(img2,axis=2) #take the mean of the R, G and B  
    img3 = np.array(img3) # img.shape: height x width x channel
    img3 = img3/255        # scaling from [0,1]
    #img3 = np.mean(img3,axis=2) #take the mean of the R, G and B 
    multi_img = np.array((img1, img2, img3))
    multi_img = multi_img[:, 20:-20, 20:-20]
    image_data[i] = multi_img.T
    print('done')
    #j += 1


for i in range(0,len(required_urls1)):
    print('i = ', i)
    arr = (image_data[i]*255).astype('uint8')
    img = Image.fromarray(arr)
#    img.save('/Users/sumi/python/research/multi_solar_images_trial/'+str(i)+'.jpg')
    i_str = str(i)
    if len(i_str) == 1:
        i_str = str(0) + str(0) + str(0) + i_str
    elif len(i_str) == 2:
        i_str = str(0) + str(0) + i_str
    elif len(i_str) == 3: 
        i_str = str(0) + i_str
    img.save('/Users/sumi/python/research/data/multi_solar_images_trial/'+i_str+'.jpg')
    print('done')
 