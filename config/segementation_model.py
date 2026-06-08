
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.optimizers import Adam
from pathlib import Path
import tensorflow as tf
import cv2 as cv
import numpy as np
from constants import RESULT


class SegmentationModel():
    def __init__(self):
        input_shape = (512, 512, 3)
        inputs = Input(input_shape)
        c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
        c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
        c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
        c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
        p4 = MaxPooling2D((2, 2))(c4)

        c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
        c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

        u6 = UpSampling2D((2, 2))(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
        c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

        u7 = UpSampling2D((2, 2))(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
        c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

        u8 = UpSampling2D((2, 2))(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
        c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

        u9 = UpSampling2D((2, 2))(c8)
        u9 = concatenate([u9, c1])
        c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
        c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

        self.model = Model(inputs=[inputs], outputs=[outputs])
        self.model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    def train_segmentation(self, train_root, annotation_root, epochs=20, batch_size=8):
        train_dataset, val_dataset = self.prepare_segmentation_data(
            train_root,
            annotation_root
        )

        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs
        )

        self.model.save("model/segmentation_model.keras")
        return history

    def prepare_segmentation_data(self, train_root, annotation_root):
        class_names = ["GUN", "knife", "shuriken"]

        image_paths, mask_paths = self.collect_image_mask_pairs(
            train_root,
            annotation_root,
            class_names
        )

        train_images, train_masks, val_images, val_masks = self.split_pairs(
            image_paths,
            mask_paths
        )

        train_dataset = self.make_dataset(train_images, train_masks, batch_size=8, training=True)
        val_dataset = self.make_dataset(val_images, val_masks, batch_size=8, training=False)

        return train_dataset, val_dataset

    def collect_image_mask_pairs(self, image_root, mask_root, class_names):
        image_paths = []
        mask_paths = []

        for class_name in class_names:
            image_dir = Path(image_root) / class_name
            mask_dir = Path(mask_root) / class_name

            if not image_dir.exists() or not mask_dir.exists():
                continue

            for image_path in sorted(image_dir.iterdir()):
                if not image_path.is_file():
                    continue

                mask_path = mask_dir / image_path.name
                if mask_path.exists():
                    image_paths.append(str(image_path))
                    mask_paths.append(str(mask_path))

        if not image_paths:
            raise ValueError("No image-mask pairs found.")

        return image_paths, mask_paths

    def collect_safe_pairs(safe_root):
        image_paths = []

        safe_dir = Path(safe_root)
        if not safe_dir.exists():
            return image_paths

        for image_path in sorted(safe_dir.iterdir()):
            if image_path.is_file():
                image_paths.append(str(image_path))

        return image_paths

    def split_pairs(self, image_paths, mask_paths, train_ratio=0.8, seed=123):
        total = len(image_paths)

        indices = tf.range(total)
        indices = tf.random.shuffle(indices, seed=seed)

        image_paths = tf.gather(tf.constant(image_paths), indices)
        mask_paths = tf.gather(tf.constant(mask_paths), indices)

        split_index = int(total * train_ratio)

        train_images = image_paths[:split_index]
        train_masks = mask_paths[:split_index]

        val_images = image_paths[split_index:]
        val_masks = mask_paths[split_index:]

        return train_images, train_masks, val_images, val_masks

    def load_image(self, image_path, image_size=(512, 512)):
        image_bytes = tf.io.read_file(image_path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, image_size, method='bilinear')
        image = tf.cast(image, tf.float32) / 255.0
        return image
    
    def load_mask(self, mask_path, image_size=(512, 512)):
        mask_bytes = tf.io.read_file(mask_path)
        mask = tf.io.decode_image(mask_bytes, channels=1, expand_animations=False)
        mask = tf.image.resize(mask, image_size, method='nearest')
        mask = tf.cast(mask, tf.float32) / 255.0
        mask = tf.where(mask > 0.5, 1.0, 0.0)
        return mask
    
    def load_pair(self, image_path, mask_path):
        image = self.load_image(image_path)
        mask = self.load_mask(mask_path)
        return image, mask

    def load_safe_pair(self, image_path, image_size=(512, 512)):
        image = self.load_image(image_path, image_size)
        mask = tf.zeros((image_size[0], image_size[1], 1), dtype=tf.float32)
        return image, mask

    def make_dataset(self, image_paths, mask_paths, batch_size=8, training=True):
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
        dataset = dataset.map(self.load_pair, num_parallel_calls=tf.data.AUTOTUNE)

        if training:
            dataset = dataset.shuffle(512, seed=123)

        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    def load_model(self):
        model_path_keras = Path("model/segmentation_model.keras")
        model_path_h5 = Path("model/segmentation_model.h5")

        if model_path_keras.exists():
            self.model = load_model(str(model_path_keras))
        elif model_path_h5.exists():
            self.model = load_model(str(model_path_h5))
        else:
            raise FileNotFoundError("No segmentation model found in model/ directory.")

        return self.model

    def segment_threat(self, image_path, threshold=0.5, save_output=True):
        image = cv.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        original_h, original_w = image.shape[:2]
        resized = cv.resize(image, (512, 512), interpolation=cv.INTER_LINEAR)
        input_tensor = resized.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0)

        prediction = self.model.predict(input_tensor, verbose=0)[0, :, :, 0]
        binary_mask = (prediction >= threshold).astype(np.uint8)
        binary_mask = cv.resize(binary_mask, (original_w, original_h), interpolation=cv.INTER_NEAREST)
        binary_mask = (binary_mask * 255).astype(np.uint8)

        overlay = image.copy()
        overlay[binary_mask > 0] = [0, 0, 255]
        blended = cv.addWeighted(image, 0.7, overlay, 0.3, 0)

        if save_output:
            
            result_annotation_path = str(Path(RESULT + '\\segment_images').with_name(f"{Path(image_path).stem}.png"))
            resulted_image_path = str(Path(RESULT + '\\result_annotation').with_name(f"{Path(image_path).stem}.png"))
            cv.imwrite(resulted_image_path, binary_mask)
            cv.imwrite(result_annotation_path, blended)

        return binary_mask, blended
