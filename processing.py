from config.segementation_model import SegmentationModel
import os
from constants import TRAIN, TRAIN_ANNOTATION
class Processing():
    # This init will act as the main process function of the class
    def __init__(self):
        self.segment_image()
        # self.classify_image()
        
    def segment_image(self):
        segment_model = SegmentationModel()
        if not os.path.exists("model/segmentation_model.pt"):
            segment_model.train_segmentation(
                train_root=TRAIN,
                annotation_root=TRAIN_ANNOTATION,
                epochs=20,
                batch_size=8
            )
        segment_model.load_model()

    def classify_image(self):
        from config.classification_model import Classification_Model
        model = Classification_Model()
        print("\nTrain: ", TRAIN)
        if not os.path.exists("model/classification_model.pt"):
            model.train(TRAIN)
        model.load_model()
        # model.predict_class(TEST)


        


    
    

        