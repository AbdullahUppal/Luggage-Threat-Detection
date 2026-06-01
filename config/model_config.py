# This class will contain the model architecture initialization, traning and then using the model for getting output
import numpy as np
import math
import os
import keras
from keras.models import *
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import *
from keras.losses import *
from keras.preprocessing.image import *

class classification_model:
    def __init__(self):    
        classes = 2
        # creating model
        inputs = Input((512, 512, 3))
        conv1 = Conv2D(8, 3, activation='relu', padding='same')(inputs)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
        conv2 = Conv2D(16, 3, activation='relu', padding='same')(pool1)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
        conv3 = Conv2D(32, 3, activation='relu', padding='same')(pool2)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2,2))(conv3)
        conv4 = Conv2D(64, 3, activation='relu', padding='same')(pool3)
        conv4 = BatchNormalization()(conv4)
        pool4 = MaxPooling2D(pool_size=(2,2))(conv4)
        conv5 = Conv2D(128, 3, activation='relu', padding='same')(pool4)
        conv5 = BatchNormalization()(conv5)
        drop5 = Dropout(0.25)(conv5)
        x = Flatten()(drop5)
        x = Dense(128, activation='relu', name='Dense_1', dtype='float32')(x)
        x = Dense(64, activation='relu', name='Dense_2', dtype='float32')(x)
        x = Dense(8, activation='relu', name='Dense_3', dtype='float32')(x)
        x = Dense(classes, activation='softmax', name='Output', dtype='float32')(x)
        my_model = Model(inputs=[inputs], outputs=[x])
        my_optimizer = Adam(lr=0.00001)
        my_model.compile(loss='categorical_crossentropy', optimizer=my_optimizer,metrics=['categorical_accuracy'])
        my_model.summary()
