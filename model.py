###Import required libraries and classes###
import numpy as np
import pandas as pd
import os
import argparse
import base64
import json
import h5py
import csv
import cv2
from skimage.exposure import adjust_gamma
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.layers import Lambda, Cropping2D
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from scipy.misc import imresize
from random import shuffle
import sklearn

###Load Data###
#load csv file, mirroring images and creating sample arrays
samples = []
with open('.\driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)



###Splitting###
#Splitting data into train and validation sets
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.20)



###Creating Generator and Performing Preprocessing on Each Batch###
#The code is written in very simple steps to be easily reviewed in the future
def generator(samples, batch_size):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            #Define assisting parameters
            image_center=[]
            image_left=[]
            image_right=[]
            angle_center=[]
            angle_left=[]
            angle_right=[]
            angle_correction=0.3

            #In this section, lists of recorded images will be created along with their
            #corresponding angles. Also, all the images will be mirrored and the corresponding angles
            #will be multiplied by -1 and added to the angle lists. For left and right cameras, the angle_correction
            #value will be added/subtracted to be able to treat left and right camera images as a center image
            
            for batch_sample in batch_samples:
                #center camera
                source_center_path=batch_sample[0]
                image=cv2.imread(source_center_path)
                image_center.append(image)
                image_center.append(cv2.flip(image,1))

                #left camera
                source_left_path=batch_sample[1]
                image=cv2.imread(source_left_path)
                image_left.append(image)
                image_left.append(cv2.flip(image,1))

                #right camera
                source_right_path=batch_sample[2]
                image=cv2.imread(source_right_path)
                image_right.append(image)
                image_right.append(cv2.flip(image,1))

                #steering angles
                angle_center.append(float(batch_sample[3]))
                angle_center.append(-1*float(batch_sample[3]))
                angle_left.append(float(batch_sample[3])+angle_correction)
                angle_left.append(-1*float(batch_sample[3])-angle_correction)
                angle_right.append(float(batch_sample[3])-angle_correction)
                angle_right.append(-1*float(batch_sample[3])+angle_correction)

            #Arrays with proper shapes are defined for images of each camera. Angle lists are converted into arrays.
            X_train_center=np.ndarray(shape=(len(angle_center), 32, 96, 3))
            X_train_left=np.ndarray(shape=(len(angle_center), 32, 96, 3))
            X_train_right=np.ndarray(shape=(len(angle_center), 32, 96, 3))
            y_train_center=np.array(angle_center)
            y_train_left=np.array(angle_left)
            y_train_right=np.array(angle_right)

            #Images will be resized to 32Hx96W to be processed faster. 
            for i in range(len(angle_center)):
                image_data_center, image_data_left, image_data_right = image_center[i], image_left[i], image_right[i]
                X_train_center[i] = imresize(image_data_center, (32,96,3))
                X_train_left[i] = imresize(image_data_left, (32,96,3))
                X_train_right[i] = imresize(image_data_right, (32,96,3))
                
            #Pre-process images and their corresponding steering angles will be combined in the same order
            X_train=np.concatenate((X_train_center, X_train_left, X_train_right), axis=0)
            y_train=np.concatenate((y_train_center, y_train_left, y_train_right), axis=0)

            #outputting pre-processed batch sample
            yield sklearn.utils.shuffle(X_train, y_train)




###Configure Model###
#Nvidia model has been used and modified accordingly
model = Sequential()

#Cropping top and bottom of images to only feed the area showing the road and avoiding distraction
model.add(Cropping2D(cropping=((12,5), (0,0)), input_shape=(32,96,3)))

#pre-process layer for value normalization
model.add(BatchNormalization(input_shape=(15,96,3), axis=1))

#CNN based on NVIDIA architecture
model.add(Conv2D(3, (3, 3), activation="relu", strides=(2, 2), padding="valid"))

model.add(Dropout(.25))

model.add(Activation('relu'))

model.add(Conv2D(24, (3, 3), activation="relu", strides=(1, 2), padding="valid"))

model.add(Dropout(.25))

model.add(Activation('relu'))

model.add(Conv2D(36, (3, 3), activation="relu", padding="valid"))

model.add(Dropout(.25))

model.add(Activation('relu'))

model.add(Conv2D(48, (2, 2), activation="relu", padding="valid"))

model.add(Activation('relu'))

model.add(Conv2D(64, (2, 2), activation="relu", padding="valid"))

model.add(Flatten())

model.add(Dense(1164))

model.add(Dropout(.25))

model.add(Activation('relu'))

model.add(Dense(100))

model.add(Dropout(.25))

model.add(Activation('relu'))

model.add(Dense(50))

model.add(Dropout(.25))

model.add(Activation('relu'))

model.add(Dense(10))

model.add(Activation('relu'))

model.add(Dense(1))

model.summary()

adam=Adam(lr=0.0001)

model.compile(loss='mse', optimizer='adam')

#Calling the generator method to be used to generate training and validation batches
train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

###Train Model###
model.fit_generator(train_generator,
                    steps_per_epoch= 1,
                    validation_data=validation_generator,
                    validation_steps=1,
                    verbose=2,
                    epochs=45)




###Output###
model_json = model.to_json()
with open('model.json', 'w') as fd:
   json.dump(model.to_json(), fd)
model.save('model.h5')

