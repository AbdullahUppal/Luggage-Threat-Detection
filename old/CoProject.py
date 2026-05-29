import cv2
import numpy as np
import os
import keras.src
import tensarflow_keras
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.losses import binary_crossentropy
from keras.src.optimizers import Adam
from tensorflow.python.keras import Sequential


def load_data(image_dir, mask_dir):
    image = []
    masks = []
    # print(len(train_img_dir), len(train_mask_dir))
    for x in range(0, len(train_img_dir)):
        img = cv2.imread(train_img_dir[x])
        mask = cv2.imread(train_mask_dir[x])
        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256))
        mask[mask > 0.5] = 1
        image.append(img)
        masks.append(mask)
    #     print(image)
    return (np.array(image), np.expand_dims(np.array(masks), axis=-1))


train_img_dir = []

# Walk through the directory tree
for root, dirs, files in os.walk("train/"):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            # Append the full path of image files to the list
            train_img_dir.append(os.path.join(root, file))

train_mask_dir = []

# Walk through the directory tree
for root, dirs, files in os.walk("annotationsTrain/"):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            # Append the full path of image files to the list
            train_mask_dir.append(os.path.join(root, file))

test_img_dir = []

# Walk through the directory tree
for root, dirs, files in os.walk("test/"):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            # Append the full path of image files to the list
            test_img_dir.append(os.path.join(root, file))

test_mask_dir = []

# Walk through the directory tree
for root, dirs, files in os.walk("annotationstest/"):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            # Append the full path of image files to the list
            test_mask_dir.append(os.path.join(root, file))

test_safe_dir = []

# Walk through the directory tree
for root, dirs, files in os.walk("safeTest/"):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            # Append the full path of image files to the list
            test_safe_dir.append(os.path.join(root, file))

train_safe_dir = []

# Walk through the directory tree
for root, dirs, files in os.walk("safeTrain/"):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            # Append the full path of image files to the list
            train_safe_dir.append(os.path.join(root, file))

train_images, train_masks = load_data(train_img_dir, train_mask_dir)
test_images, test_masks = load_data(test_img_dir, test_mask_dir)

# Evaluate the model (optional)
# You can use metrics like Dice coefficient or Jaccard index for segmentation evaluation

datagen = ImageDataGenerator(rotation_range=20, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
datagen.fit(train_images)

# Define image dimensions (assuming consistent size)
img_height, img_width, channels = train_images[0].shape

# Data augmentation (optional)
datagen = ImageDataGenerator(rotation_range=20, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
datagen.fit(train_images)

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

# Compile the model
model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model
model.fit(datagen.flow(train_images, train_masks[:, :, :, None]),
          epochs=10, validation_data=(test_images, test_masks[:, :, :, None]))

# Evaluate the model (optional)
# You can use metrics like accuracy, precision, recall, F1-score for evaluation

# Make predictions on new images
predictions = model.predict(test_images)
for i, prediction in enumerate(predictions):
  predicted_class = int(round(prediction[0]))  # Threshold for class
  print(f"Image {i+1} classified as: {predicted_class}")  # 0: Safe, 1: Threat
