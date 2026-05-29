import os
import cv2
import numpy as np
from keras import Sequential, Input, Model
from keras.src import layers
from keras.src.losses import binary_crossentropy
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf



# Data loading function
def load_data(image_dir, mask_dir=None, img_size=(256, 256)):
    images = []
    masks = []

    for img_path in image_dir:
        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size)
        images.append(img)

        if mask_dir:
            mask_path = img_path.replace("train", "annotationsTrain").replace("test", "annotationstest")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, img_size)
            masks.append(mask)

    images = np.array(images)
    if masks:
        masks = np.array(masks)
        masks = masks[..., np.newaxis] / 255.0

    return images, masks if masks else images


# Load datasets
train_img_dir = [os.path.join(dp, f) for dp, dn, fn in os.walk("train/") for f in fn if
                 f.lower().endswith((".jpg", ".jpeg", ".png"))]
train_safe_dir = [os.path.join(dp, f) for dp, dn, fn in os.walk("safeTrain/") for f in fn if
                  f.lower().endswith((".jpg", ".jpeg", ".png"))]
test_img_dir = [os.path.join(dp, f) for dp, dn, fn in os.walk("test/") for f in fn if
                f.lower().endswith((".jpg", ".jpeg", ".png"))]
test_safe_dir = [os.path.join(dp, f) for dp, dn, fn in os.walk("safeTest/") for f in fn if
                 f.lower().endswith((".jpg", ".jpeg", ".png"))]

train_images, train_masks = load_data(train_img_dir)
train_safe_images = load_data(train_safe_dir)[0]
test_images, test_masks = load_data(test_img_dir)
test_safe_images = load_data(test_safe_dir)[0]

# Combine threat and safe images for classification
train_images_combined = np.concatenate((train_images, train_safe_images), axis=0)
train_labels_combined = np.concatenate((np.ones(len(train_images)), np.zeros(len(train_safe_images))), axis=0)
test_images_combined = np.concatenate((test_images, test_safe_images), axis=0)
test_labels_combined = np.concatenate((np.ones(len(test_images)), np.zeros(len(test_safe_images))), axis=0)

# Normalize images
train_images_combined = train_images_combined / 255.0
test_images_combined = test_images_combined / 255.0


# Define classification model
def build_classification_model(input_shape):
    model = Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss=binary_crossentropy, metrics=['accuracy'])
    return model


# Define U-Net model for segmentation
def build_unet_model(input_shape):
    inputs = Input(input_shape)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    u6 = layers.UpSampling2D((2, 2))(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.UpSampling2D((2, 2))(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.UpSampling2D((2, 2))(c7)
    u8 =layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.UpSampling2D((2, 2))(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Build and train classification model
classification_model = build_classification_model((256, 256, 3))
classification_model.fit(train_images_combined, train_labels_combined, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate classification model
pred_labels_combined = (classification_model.predict(test_images_combined) > 0.5).astype("int32")
print(f"Classification Accuracy: {accuracy_score(test_labels_combined, pred_labels_combined)}")
print(f"Confusion Matrix:\n {confusion_matrix(test_labels_combined, pred_labels_combined)}")

# Build and train U-Net model
unet_model = build_unet_model((256, 256, 3))
unet_model.fit(train_images, train_masks, epochs=10, batch_size=16, validation_split=0.1)

# Evaluate U-Net model
pred_masks = unet_model.predict(test_images)
dice_scores = []
for i in range(len(test_masks)):
    intersection = np.sum(pred_masks[i] * test_masks[i])
    union = np.sum(pred_masks[i]) + np.sum(test_masks[i])
    dice_score = (2. * intersection + 1) / (union + 1)
    dice_scores.append(dice_score)
print(f"Mean Dice Score: {np.mean(dice_scores)}")

# Save models
classification_model.save('classification_model.h5')
unet_model.save('unet_model.h5')

# Save some example results
for i in range(5):
    cv2.imwrite(f'results/pred_mask_{i}.png', (pred_masks[i] * 255).astype(np.uint8))
    cv2.imwrite(f'results/test_image_{i}.png', (test_images[i] * 255).astype(np.uint8))
    cv2.imwrite(f'results/test_mask_{i}.png', (test_masks[i] * 255).astype(np.uint8))