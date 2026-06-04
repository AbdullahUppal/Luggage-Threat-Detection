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
import os
import numpy as np


class Classification_Model:
    def __init__(self):
        if not os.path.exists("/model/classification_model.h5"):
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
        else:
            self.my_model=self.load_model()
    
    def load_model(self):
       self.model = load_model('model/classification_model.h5')
       return self.model

    def train(self, train_path):
        dataset_dir = train_path

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
        _ = self.my_model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=10 # You can increase this as needed
        )

        self.my_model.save('model/classification_model.h5')

    def predict_class(self, IMAGE_PATH):

        # --- 2. SETUP THE TEST DATASET ---
        # Point this to a folder containing your new, unseen images.
        test_dir = IMAGE_PATH 

        print("Preparing data pipeline...")
        test_dataset = image_dataset_from_directory(
            test_dir,
            labels=None,            # Critical: Set to None because these are new images being tested
            shuffle=False,          # Critical: Keep order strict so predictions match filenames
            image_size=(512, 512),  # Must perfectly match your Input layer
            batch_size=50          # The GPU will process exactly 16 images at a time to save memory
        )

        # --- 3. RUN THE BATCH PREDICTION ---
        # Keras handles the batch loop automatically behind the scenes
        print("Running predictions on all images...")
        predictions = self.my_model.predict(test_dataset)

        # --- 4. MATCH PREDICTIONS TO FILENAMES ---
        # Get the exact order of files that the dataset loaded
        file_paths = test_dataset.file_paths 
        class_names = ['safe', 'threat'] # Must match the alphabetical order of your training folders

        print("\n--- RESULTS ---")
        # Loop through the results (we will just print the first 20 so it doesn't flood your console)
        for i in range(0,(file_paths)):
            
            # Extract just the filename from the full path
            file_name = os.path.basename(file_paths[i])
            
            # Figure out the highest probability class
            predicted_index = np.argmax(predictions[i])
            predicted_class = class_names[predicted_index]
            confidence = np.max(predictions[i]) * 100
            
            print(f"Image: {file_name} | Prediction: {predicted_class} | Confidence: {confidence:.2f}%")
        print(f"\nSuccessfully processed {len(file_paths)} images!")