import cv2
import numpy as np

# from keras import Input, Model
# from keras.src.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
# from keras.src.optimizers import Adam
# from sklearn.utils  import shuffle
# from sklearn.metrics  import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt
# import ipyplot
# import tensorflow as tf
# import math
import os



def load_data(image_dir, mask_dir):
    image = []
    masks = []
    for x in range(0, len(image_dir)):
        img = cv2.imread(image_dir[x])
        mask = cv2.imread(mask_dir[x])
        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256))
        mask[mask > 0.5] = 1
        image.append(img)
        masks.append(mask)
    #     print(image)
    return (image, masks)

train_img_dir = []

for root, dirs, files in os.walk("train/"):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            # Append the full path of image files to the list
            train_img_dir.append(os.path.join(root, file))

train_mask_dir = []

for root, dirs, files in os.walk("annotationsTrain/"):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            # Append the full path of image files to the list
            train_mask_dir.append(os.path.join(root, file))

test_img_dir = []

for root, dirs, files in os.walk("test/"):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            # Append the full path of image files to the list
            test_img_dir.append(os.path.join(root, file))

test_mask_dir = []

for root, dirs, files in os.walk("annotationstest/"):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            # Append the full path of image files to the list
            test_mask_dir.append(os.path.join(root, file))

test_safe_dir = []

for root, dirs, files in os.walk("safeTest/"):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            # Append the full path of image files to the list
            test_safe_dir.append(os.path.join(root, file))

train_safe_dir = []

for root, dirs, files in os.walk("safeTrain/"):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            # Append the full path of image files to the list
            train_safe_dir.append(os.path.join(root, file))

train_images, train_masks = load_data(train_img_dir, train_mask_dir)
test_images, test_masks = load_data(test_img_dir, test_mask_dir)



# print(" Training Shape :", train_images., train_masks.shape)
# print(" Testing Shape :", test_images.shape, test_masks.shape)

# classes = 2
# # creating model
# inputs = Input(256, 256, 3)
# conv1 = Conv2D(8, 3, activation='relu', padding='same')(inputs)
# conv1 = BatchNormalization()(conv1)
# pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
# conv2 = Conv2D(16, 3, activation='relu', padding='same')(pool1)
# conv2 = BatchNormalization()(conv2)
# pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
# conv3 = Conv2D(32, 3, activation='relu', padding='same')(pool2)
# conv3 = BatchNormalization()(conv3)
# pool3 = MaxPooling2D(pool_size=(2,2))(conv3)
# conv4 = Conv2D(64, 3, activation='relu', padding='same')(pool3)
# conv4 = BatchNormalization()(conv4)
# pool4 = MaxPooling2D(pool_size=(2,2))(conv4)
# conv5 = Conv2D(128, 3, activation='relu', padding='same')(pool4)
# conv5 = BatchNormalization()(conv5)
# drop5 = Dropout(0.25)(conv5)
# x = Flatten()(drop5)
# x = Dense(128, activation='relu', name='Dense_1', dtype='float32')(x)
# x = Dense(64, activation='relu', name='Dense_2', dtype='float32')(x)
# x = Dense(8, activation='relu', name='Dense_3', dtype='float32')(x)
# x = Dense(classes, activation='softmax', name='Output', dtype='float32')(x)
# my_model = Model(inputs=[inputs], outputs=[x])
# my_optimizer = Adam()
# my_model.compile(loss='categorical_crossentropy', optimizer=my_optimizer,metrics=['categorical_accuracy'])
# my_model.summary()
#
# my_model_ = my_model.fit(x=train_images[0], y=train_masks[0], batch_size=600, epochs=10)
#
# my_predictions = my_model.predict(test_images[0])
# print(my_predictions.shape)
# classes = ['safe', 'threat']
# cmd = ConfusionMatrixDisplay((confusion_matrix(list(np.argmax(test_masks[0],axis=1)), list(np.argmax(my_predictions, axis=1)))),
# display_labels=classes)
# cmd.plot()
# plt.show()
#
