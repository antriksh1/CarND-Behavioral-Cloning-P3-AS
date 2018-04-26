import os
import sys
import csv
import cv2
import numpy as np
import sklearn
import gc

# Training folder that I used to train - combination of provided 'data' and my training set
training_folder = 'data_training3'

# Reading in the log
samples = []
with open(training_folder + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
                samples.append(line)

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# training - validation split of 70% - 30%
train_samples, validation_samples = train_test_split(samples, test_size=0.3)

# Generator to provide the data to the model
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = training_folder + '/IMG/' + batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    angle = float(batch_sample[3])
                    if(i == 1): # left image
                        angle = angle + 0.2
                    if(i == 2): # right image
                        angle = angle - 0.2
                    images.append(image)
                    angles.append(angle)
                    images.append(cv2.flip(image, 1))
                    angles.append(angle*-1)
            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
            
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# After trying the NVIDIA model + another model I found on the web, and 2 variations of LeNet, 
# this is the modified version of LeNet I chose, because it gave me the smoothest driving

# LeNet
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,25),(0,0))))
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(16,3,3,activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(80))
model.add(Dense(40))
model.add(Dense(1))

# Generating and saving the model
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=6*len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=8)
model.save('model.h5.' + training_folder)
print("Model Saved!")

# Garbage-Collection - to prevent memory leaks
gc.collect()
