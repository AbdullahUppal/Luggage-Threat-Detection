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
        has_keras = os.path.exists("model/segmentation_model.keras")
        has_h5 = os.path.exists("model/segmentation_model.h5")

        if not has_keras and not has_h5:
            segment_model.train_segmentation(
                train_root=TRAIN,
                annotation_root=TRAIN_ANNOTATION,
                epochs=20,
                batch_size=2
            )
        segment_model.load_model()
        filepaths = self._getfilepaths(TEST)

        for image_path in filepaths:
            binary_mask, segmented_image = segment_model.segment_threat(image_path)

    def classify_image(self):
        model = Classification_Model()
        print("\nTrain Annotation: ", TRAIN_ANNOTATION)
        if not os.path.exists("model/classification_model.h5"):
            model.train(TRAIN_ANNOTATION)
        model.predict_class(RESULT)


    def _getfilepaths(self, target_directory):
            if not target_directory:
                return []

            target_path = Path(target_directory)
            if not target_path.exists():
                return []

            file_paths = []
            # os.walk automatically enters every subfolder it finds
            for root, folders, files in os.walk(str(target_path)):
                for file in files:
                    # os.path.join combines the folder path and file name safely
                    full_path = os.path.join(root, file)
                    file_paths.append(full_path)  
            return file_paths


    
    

        