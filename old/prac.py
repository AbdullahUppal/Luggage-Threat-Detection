import os
import cv2
# from keras.src.layers import Flatten
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, concatenate
# from tensorflow.keras.models import Model
# from tensorflow.keras.losses import binary_crossentropy
# # from tensorflow.keras.metrics import DiceCoefficient
#
# # Additional libraries for data manipulation and evaluation metrics
# import numpy as np
# from sklearn.metrics import confusion_matrix
#
# gun = os.listdir("train/GUN")
# knife = os.listdir("train/knife")
# shuriken = os.listdir("train/shuriken")
# safe = os.listdir("train/safe")
#
# # Masks
# gunMask = os.listdir("train/annotations/GUN")
# knifeMask = os.listdir("train/annotations/knife")
# shurikenMask = os.listdir("train/annotations/shuriken")
#
# # Load pre-trained VGG16 model (without top layers)
# base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
#
# # Freeze base model layers (optional)
# for layer in base_model.layers:
#     layer.trainable = False
#
# # Add classification head on top of pre-trained features
# x = base_model.output
# x = Flatten()(x)
# x = Dense(units=1, activation="sigmoid")(x)  # Sigmoid for binary classification
#
# classification_model = Model(inputs=base_model.input, outputs=x)
#
# # Compile classification model
# classification_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
#
#
#
# for x in range(0, len(gun)):
#     img = cv.imread("train/GUN/" + gun[x])
#     img1 = img[:, :, 2]
#
# print("done")
#
#
#
#
# ########## Part_2
#
#
# import cv2
# from skimage.feature import local_binary_pattern
#
# def get_lbp_features(image, radius=8, points=8):
#   """Extracts LBP features from an image."""
#   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
#   lbp = local_binary_pattern(gray, points, radius)
#   return lbp.ravel()  # Flatten the LBP image into a feature vector
#
#
# def train_classifier(train_images, train_labels):
#   """Trains a classifier using LBP features."""
#   lbp_features = []
#   for image in train_images:
#     lbp_features.append(get_lbp_features(image))
#
#   # Choose a classifier (e.g., Support Vector Machine)
#   from sklearn.svm import SVC
#   clf = SVC(kernel='linear')  # Linear kernel for efficiency
#   clf.fit(lbp_features, train_labels)
#   return clf
#
#
# def predict_class(image, clf):
#   """Predicts the class (safe or threat) of an image."""
#   lbp_features = get_lbp_features(image)
#   prediction = clf.predict(lbp_features.reshape(1, -1))
#   return prediction[0]  # Return the predicted class label
#
#
# # Load your training data (images and labels)
# # Replace with your data loading logic
# train_images = []
# train_labels = []  # 0: Safe, 1: Threat
#
# # Train the classifier
# clf = train_classifier(train_images, train_labels)
#
# # Test on a new image
# test_image = cv2.imread("path/to/test_image.jpg")
# prediction = predict_class(test_image, clf)
#
# if prediction == 0:
#   print("Image classified as Safe")
# else:
#   print("Image classified as Threat")


# image_files = []
#
# # Walk through the directory tree
# for root, dirs, files in os.walk("train"):
#     for file in files:
#         if file.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
#             # Append the full path of image files to the list
#             image_files.append(os.path.join(root, file))
#
# # Now 'image_files' contains all image file paths
# print(image_files)
#
# img = cv.imread("annotations/GUN/B0012_0001.png",0)
# cv.imshow("orig",img)
# for x in range(0,img.shape[0]):
#     for y in range(0,img.shape[1]):
#         if (img[x][y]!=0):
#             img[x][y]=255
#
# cv.imshow("annotation",img)
# cv.waitKey()

def identify(images,masks):
    detected_img = []
    for i in range(0,len(images)):
        cv2.imshow("img", images)
        ret, thresh = cv2.threshold(images, 80, 255, cv2.THRESH_BINARY)
        masks[masks > 0.5] = 255
        thresh = 255-thresh
        print(thresh.shape)
        cv2.imshow("thresh", thresh)
        cv2.imshow("mask", masks)
        candidate_regions = cv2.bitwise_and(thresh, masks)
        cv2.imshow("candidate", candidate_regions)
        cv2.waitKey()

        detected_img.append(candidate_regions)
    return detected_img


identify(cv2.imread("test/gun/B0011_0001.png")[:,:,2],cv2.imread("annotationstest/gun/B0011_0001.png",0))