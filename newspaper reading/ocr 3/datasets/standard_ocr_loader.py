import os
from typing import List
from .base_loader import BaseOCRDatasetLoader, OCRItem

class StandardOCRDatasetLoader(BaseOCRDatasetLoader):
    """
    Loader for the Standard OCR Dataset (Kaggle format):
    Directory structure:
      <dataset_dir>/
        training_data/
          <class_label>/
            <image_id>.png
        testing_data/
          <class_label>/
            <image_id>.png
    """

    def __init__(self):
        super().__init__(name="Standard OCR Dataset")

    def load(self, data_dir: str, split_name: str = "training_data") -> List[OCRItem]:
        items = []
        target_dir = os.path.join(data_dir, split_name) if os.path.exists(os.path.join(data_dir, split_name)) else data_dir

        if not os.path.exists(target_dir):
            return items

        for entry in os.listdir(target_dir):
            class_path = os.path.join(target_dir, entry)
            if not os.path.isdir(class_path):
                continue

            transcription = entry
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                    img_path = os.path.join(class_path, fname)
                    items.append(OCRItem(
                        image_path=img_path,
                        transcription=transcription,
                        dataset_source="Standard OCR Dataset",
                        metadata={"class": entry, "split": split_name}
                    ))

        return items
