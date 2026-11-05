import os
import hashlib
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import cv2

@dataclass
class OCRItem:
    """Standardized image-text paired sample across all supported OCR datasets."""
    image_path: str
    transcription: str
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    dataset_source: str = "custom"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self, min_width: int = 8, min_height: int = 8) -> bool:
        if not self.image_path or not os.path.exists(self.image_path):
            return False
        if not self.transcription or not self.transcription.strip():
            return False
        return True

@dataclass
class DatasetSplit:
    train: List[OCRItem]
    val: List[OCRItem]
    test: List[OCRItem]
    real_world_test: List[OCRItem] = field(default_factory=list)

    @property
    def total_samples(self) -> int:
        return len(self.train) + len(self.val) + len(self.test) + len(self.real_world_test)

    def summary(self) -> Dict[str, int]:
        return {
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test),
            "real_world_test": len(self.real_world_test),
            "total": self.total_samples
        }

class BaseOCRDatasetLoader:
    def __init__(self, name: str):
        self.name = name

    def load(self, data_dir: str) -> List[OCRItem]:
        raise NotImplementedError

    def validate_and_filter(self, items: List[OCRItem]) -> Tuple[List[OCRItem], List[str]]:
        """Filters missing or corrupted images and returns valid items + error log."""
        valid_items = []
        errors = []

        for item in items:
            if not item.is_valid():
                errors.append(f"Invalid record: {item.image_path}")
                continue

            # Check image integrity
            try:
                img = cv2.imread(item.image_path, cv2.IMREAD_GRAYSCALE)
                if img is None or img.size == 0 or img.shape[0] < 6 or img.shape[1] < 6:
                    errors.append(f"Corrupted image payload: {item.image_path}")
                    continue
                valid_items.append(item)
            except Exception as e:
                errors.append(f"Exception reading {item.image_path}: {str(e)}")

        return valid_items, errors

    @staticmethod
    def create_leak_free_splits(
        items: List[OCRItem],
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10
    ) -> DatasetSplit:
        """
        Uses cryptographic hash of the image path or parent document to guarantee zero data leakage
        between train, val, and test splits across runs.
        """
        train, val, test = [], [], []

        for item in items:
            # Hash path to assign bucket deterministically [0, 1000)
            h = int(hashlib.md5(item.image_path.encode('utf-8')).hexdigest()[:8], 16) % 1000
            val_thresh = int(train_ratio * 1000)
            test_thresh = int((train_ratio + val_ratio) * 1000)

            if h < val_thresh:
                train.append(item)
            elif h < test_thresh:
                val.append(item)
            else:
                test.append(item)

        return DatasetSplit(train=train, val=val, test=test)
