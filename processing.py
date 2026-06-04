import cv2 as cv
from config.classification_model import Classification_Model
import os
from constants import TRAIN_CLASSIFICATION, TEST
class Processing():
    # This init will act as the main process function of the class
    def __init__(self):
        self.classify_image()
        
    def classify_image(self):
        model = Classification_Model()
        if not os.path.exists("model/classification_model.h5"):
            model.train(TRAIN_CLASSIFICATION)
        model.load_model()
        model.predict_class(TEST)

        


    
    

        