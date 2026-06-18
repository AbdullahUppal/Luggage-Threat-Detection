import os
from constants import TRAIN, TRAIN_ANNOTATION, TEST, RESULT
import cv2 as cv
from config import SegmentationModel, Classification_Model
class Processing():
    # This init will act as the main process function of the class
    def __init__(self):
        self.segment_image()
        # self.classify_image()
        
    def segment_image(self):
        segment_model = SegmentationModel()

        # Use trained model only; do not train here
        if not os.path.exists(segment_model.model_path):
            segment_model.train_segmentation(
                train_root=TRAIN,
                annotation_root=TRAIN_ANNOTATION,
                epochs=20,
                batch_size=8
            )

        segment_model.load_model()

        filepaths = self._getfilepaths(os.getcwd() + "\\DIP Data Upload\\test\\")
        os.makedirs(os.getcwd() + "\\DIP Data Upload\\result\\", exist_ok=True)

        for image_path in filepaths:
            output_name = os.path.splitext(os.path.basename(image_path))[0] + "_mask.png"
            output_path = os.path.join(os.getcwd(), "DIP Data Upload", "result", output_name)
            image = segment_model.segment_threat(filepath=image_path, save_path=output_path)
            

    def classify_image(self):
        classification_model = Classification_Model()
        print("\nTrain: ", TRAIN_ANNOTATION)
        if not os.path.exists("model/classification_model.pt"):
            classification_model.train(TRAIN_ANNOTATION)
        classification_model.load_model()
        classification_model.predict_class(RESULT)


    def _getfilepaths(self, target_directory):
            file_paths = []
            # os.walk automatically enters every subfolder it finds
            for root, folders, files in os.walk(target_directory):
                for file in files:
                    # os.path.join combines the folder path and file name safely
                    full_path = os.path.join(root, file)
                    file_paths.append(full_path)  
            return file_paths



        


    
    

        