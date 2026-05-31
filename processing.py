import cv2 as cv
from constants import *

class Processing():

    # This init will act as the main process function of the class
    def __init__(self):
        self.load_train_paths()

    def load_train_paths(self):
        tr_knife = os.listdir(TRAIN_KNIFE)
        tr_knife_ann = os.listdir(TRAIN_KNIFE_ANNOTATION)
        tr_gun = os.listdir(TRAIN_GUN)
        tr_gun_ann = os.listdir(TRAIN_GUN_ANNOTATION)
        tr_shuriken = os.listdir(TRAIN_SHURIKEN)
        tr_shuriken_ann = os.listdir(TRAIN_SHURIKEN_ANNOTATION)

        tr_safe = os.listdir(TRAIN_SAFE)
    

        