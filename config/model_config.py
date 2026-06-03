# This class will contain the model architecture initialization, traning and then using the model for getting output
# import numpy as np
# import math
# import os
from keras.utils import image_dataset_from_directory
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
# from keras.losses import *
# from keras.preprocessing.image import *
from tensorflow import data

from constants import TRAIN_CLASSIFICATION


class Classification_Model:
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
        self.my_model = Model(inputs=[inputs], outputs=[x])
        my_optimizer = Adam(learning_rate=0.00001)
        self.my_model.compile(loss='categorical_crossentropy', optimizer=my_optimizer, metrics=['categorical_accuracy'])
        # my_model.summary()
    
    def load_model(self):
       self.model = load_model('model/classification_model.h5')
       return self.model
    
    
    def save_model(self):
        self.my_model.save('model/classification_model.h5')

    def train(self):

        dataset_dir = TRAIN_CLASSIFICATION

        # 2. Load the training data
        # We use a validation split to keep 20% of the data for testing
        train_dataset = image_dataset_from_directory(
            dataset_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(512, 512), 
            batch_size=16,         
            label_mode='categorical'
        )

        # 3. Load the validation data
        validation_dataset = image_dataset_from_directory(
            dataset_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(512, 512),
            batch_size=16,
            label_mode='categorical'
        )

        # 4. Optimize dataset performance for training
        AUTOTUNE = data.AUTOTUNE
        train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

        # 5. Train the model
        # Assuming your class is named MyModelClass and initialized as `model_instance`
        history = self.my_model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=10 # You can increase this as needed
        )

        import matplotlib.pyplot as plt

        # Extract the data from the history object
        metrics = history.history

        # --- 1. Plot Accuracy ---
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(metrics['categorical_accuracy'], label='Training Accuracy', color='blue')
        plt.plot(metrics['val_categorical_accuracy'], label='Validation Accuracy', color='orange')
        plt.title('Model Accuracy Over Time')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()

        # --- 2. Plot Loss ---
        plt.subplot(1, 2, 2)
        plt.plot(metrics['loss'], label='Training Loss', color='blue')
        plt.plot(metrics['val_loss'], label='Validation Loss', color='orange')
        plt.title('Model Loss Over Time')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()

        plt.show()

        self.save_model()
