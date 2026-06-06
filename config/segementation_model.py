

from keras.models import Model, load_model
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, concatenate, Dropout, Flatten, Dense
from keras.optimizers import Adam

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

    def train_segmentation(self):
        pass

    def segment_threat(Self):
        pass