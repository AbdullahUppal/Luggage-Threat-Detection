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
import tensorflow as tf
import os
import numpy as np
import cv2 as cv
from pathlib import Path

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass


class Classification_Model:
    def __init__(self):
        if not os.path.exists("model/classification_model.h5"):
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
        
        dataset_dir = {
            "threat": self._getfilepaths(train_path), 
            "safe": self._getsafepath(train_path)
            }
        
        self.inner_train(dataset_dir)

    
    def inner_train(self, dataset_dir):
        class_names = ["safe", "threat"]   # keep fixed class order
        class_to_idx = {name: i for i, name in enumerate(class_names)}

        image_paths = []
        labels = []

        for class_name, paths in dataset_dir.items():
            if class_name not in class_to_idx:
                continue
            image_paths.extend(paths)
            labels.extend([class_to_idx[class_name]] * len(paths))

        if len(image_paths) == 0:
            raise ValueError("No images found in dataset_dir")

        image_paths = np.array([str(p) for p in image_paths], dtype=str)
        labels = np.array(labels, dtype=np.int32)

        # shuffle
        idx = np.arange(len(image_paths))
        np.random.shuffle(idx)
        image_paths = image_paths[idx]
        labels = labels[idx]

        # split
        split = int(0.8 * len(image_paths))
        train_paths, val_paths = image_paths[:split], image_paths[split:]
        train_labels, val_labels = labels[:split], labels[split:]

        def load_pair(path, label):
            img = tf.io.read_file(path)
            img = tf.io.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, (512, 512))
            img = tf.cast(img, tf.float32) / 255.0
            y = tf.one_hot(label, depth=len(class_names))
            return img, y

        train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        train_ds = train_ds.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.shuffle(32).batch(4).prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
        val_ds = val_ds.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.batch(4).prefetch(tf.data.AUTOTUNE)

        self.my_model.fit(train_ds, validation_data=val_ds, epochs=50, verbose=2)

        Path("model").mkdir(parents=True, exist_ok=True)
        self.my_model.save("model/classification_model.h5")
        

    def _getfilepaths(self, target_directory):
            if not target_directory:
                return []

            target_path = Path(target_directory)
            if not target_path.exists():
                return []

            file_paths = []
            # os.walk automatically enters every subfolder it finds
            for root, folders, files in os.walk(str(target_path)):
                for file in files:
                    # os.path.join combines the folder path and file name safely
                    full_path = os.path.join(root, file)
                    file_paths.append(full_path)  
            return file_paths

    def _getsafepath(self, target_directory):
        
        target_path = Path(target_directory) / "safe"
        
        file_paths = []
        width = height = 512
        target_path.mkdir(parents=True, exist_ok=True)

        if not list(target_path.iterdir()):
            for x in range(0,800):
                black_image = np.zeros((width, height), dtype=np.uint8)

                blob_count = 0
                make_random = False
                if x > 600:
                    blob_count = np.random.randint(2, 8)
                elif x > 500:
                    make_random = True
                elif x > 400:
                    blob_count = 1

                if make_random:
                    pixel_count = np.random.randint(300, 2000)
                    ys = np.random.randint(0, height, size=pixel_count)
                    zs = np.random.randint(0, width, size=pixel_count)
                    black_image[ys, zs] = 255

                for _ in range(blob_count):
                    y = np.random.randint(0, height)
                    z = np.random.randint(0, width)

                    walk_length = np.random.randint(100, 400)

                    for _ in range(walk_length):
                        black_image[y, z] = 255

                        direction = np.random.randint(0, 4)
                        if direction == 0:
                            y = max(0, y - 1)
                        elif direction == 1:
                            y = min(height - 1, y + 1)
                        elif direction == 2:
                            z = max(0, z - 1)
                        else:
                            z = min(width - 1, z + 1)
                file_name = f"safe_{x}.png"
                path = Path(target_directory) / 'safe' / file_name
                file_paths.append(path)
                cv.imwrite(path, black_image)        
        else:
            for root, folders, files in os.walk(str(target_path)):
                for file in files:
                    full_path = os.path.join(root, file)
                    file_paths.append(full_path)

        return file_paths

    def predict_class(self, IMAGE_PATH):

        # --- 2. SETUP THE TEST DATASET ---
        # Point this to a folder containing your new, unseen images.
        test_dir = Path(IMAGE_PATH) / 'result_annotation'

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
        for i in range(0,len(file_paths)):
            
            # Extract just the filename from the full path
            file_name = os.path.basename(file_paths[i])
            
            # Figure out the highest probability class
            predicted_index = np.argmax(predictions[i])
            predicted_class = class_names[predicted_index]
            confidence = np.max(predictions[i]) * 100
            
            print(f"Image: {file_name} | Prediction: {predicted_class} | Confidence: {confidence:.2f}%")
        print(f"\nSuccessfully processed {len(file_paths)} images!")