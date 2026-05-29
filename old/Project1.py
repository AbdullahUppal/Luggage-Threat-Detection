import cv2 as cv
import numpy as np
import os
from skimage.feature import local_binary_pattern

def get_lbp_features(image, radius=8, points=8):
      gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # Convert to grayscale
      lbp = local_binary_pattern(gray, points, radius)
      return lbp.ravel()  # Flatten the LBP image into a feature vector

def train_classifier(train_images, train_labels):
    lbp_features = []
    for image in train_images:
        img = cv.imread(image)
        lbp_features.append(get_lbp_features(img))
        print("doing")
    # Choose a classifier (e.g., Support Vector Machine)
    print("lbp_feature_done")
    from sklearn.svm import SVC
    clf = SVC(kernel='linear')  # Linear kernel for efficiency
    clf.fit(lbp_features, train_labels)
    return clf

def predict_class(image, clf):
    """Predicts the class (safe or threat) of an image."""
    lbp_features = get_lbp_features(image)
    prediction = clf.predict(lbp_features.reshape(1, -1))
    return prediction[0]  # Return the predicted class label


# gun = os.listdir("train/GUN")
# knife = os.listdir("train/knife")
# shuriken = os.listdir("train/shuriken")
# safe = os.listdir("train/safe")

train_img = []

# Walk through the directory tree
for root, dirs, files in os.walk("train"):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            # Append the full path of image files to the list
            train_img.append(os.path.join(root, file))

# Now 'image_files' contains all image file paths
# print(train_img)

# Masks
gunMask = os.listdir("annotations/GUN")
knifeMask = os.listdir("annotations/knife")
shurikenMask = os.listdir("annotations/shuriken")

# for x in range(0, len(gun)):
#     img = cv.imread("train/GUN/" + gun[x])
#     img1 = img[:, :, 2]
    # cv.imshow("orig", img1)
    # cv.imshow("test", get_masked(img))
    # ret, thresh = cv.threshold(img1, 80, 255, cv.THRESH_BINARY)
    # cv.imshow("er1", thresh)
    # cv.waitKey()


# Load your training data (images and labels)
# Replace with your data loading logic
 # 0: Safe, 1: Threat

# Train the classifier
clf = train_classifier(train_img, train_labels)

# Test on a new image
test_image = cv.imread("students_data/test/GUN/B0012_0001.png")
prediction = predict_class(test_image, clf)

if prediction == 0:
  print("Image classified as Safe")
else:
  print("Image classified as Threat")



