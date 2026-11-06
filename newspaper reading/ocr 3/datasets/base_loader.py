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

    @property
    def source_document_id(self) -> str:
        """Returns the document ID or falls back to parent directory/file base."""
        if "source_document_id" in self.metadata:
            return str(self.metadata["source_document_id"])
        if "publisher_id" in self.metadata:
            return str(self.metadata["publisher_id"])
        # Heuristic fallback: use parent folder or base image name
        return os.path.basename(os.path.dirname(self.image_path)) or os.path.basename(self.image_path)

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
        test_ratio: float = 0.10,
        group_key: Optional[str] = "source_document_id"
    ) -> DatasetSplit:
        """
        Guarantees zero data leakage by grouping by document/source ID before assigning splits.
        All lines/crops from a single page or document remain strictly in the same split.
        """
        train, val, test = [], [], []

        # Determine split threshold markers in [0, 1000)
        val_thresh = int(train_ratio * 1000)
        test_thresh = int((train_ratio + val_ratio) * 1000)

        # Cache group assignments
        group_buckets: Dict[str, str] = {}

        for item in items:
            if group_key == "source_document_id":
                grp = item.source_document_id
            elif group_key and group_key in item.metadata:
                grp = str(item.metadata[group_key])
            else:
                grp = item.image_path

            if grp not in group_buckets:
                h = int(hashlib.md5(grp.encode('utf-8')).hexdigest()[:8], 16) % 1000
                if h < val_thresh:
                    group_buckets[grp] = "train"
                elif h < test_thresh:
                    group_buckets[grp] = "val"
                else:
                    group_buckets[grp] = "test"

            assigned_split = group_buckets[grp]
            if assigned_split == "train":
                train.append(item)
            elif assigned_split == "val":
                val.append(item)
            else:
                test.append(item)

        return DatasetSplit(train=train, val=val, test=test)
