import os
import json
from typing import List, Dict, Any, Optional, Tuple
from .base_loader import OCRItem, DatasetSplit, BaseOCRDatasetLoader
from .standard_ocr_loader import StandardOCRDatasetLoader
from .page_xml_parser import PageXMLParser
from .enp_loader import ENPDatasetLoader

class DatasetManager:
    """
    Centralized Dataset Manager supporting:
    1. Standard OCR Dataset
    2. IMPACT Newspaper Ground Truth (PAGE XML)
    3. ENP Dataset (ICDAR / ALTO)
    4. PRImA Datasets (PAGE XML)
    + Dedicated Real-World Newspaper Evaluation Split.
    """

    def __init__(self):
        self.loaders = {
            "standard_ocr": StandardOCRDatasetLoader(),
            "impact": PageXMLParser(name="IMPACT Newspaper Ground Truth"),
            "prima": PageXMLParser(name="PRImA Layout Dataset"),
            "enp": ENPDatasetLoader()
        }

    def load_dataset(self, dataset_type: str, data_dir: str, **kwargs) -> List[OCRItem]:
        """Loads a dataset by key: 'standard_ocr', 'impact', 'prima', or 'enp'."""
        loader = self.loaders.get(dataset_type)
        if not loader:
            raise ValueError(f"Unknown dataset type: {dataset_type}. Available: {list(self.loaders.keys())}")
        return loader.load(data_dir, **kwargs)

    def validate_and_sanitize(self, items: List[OCRItem]) -> Tuple[List[OCRItem], List[str]]:
        """Filters corrupted image files and empty transcriptions."""
        valid = []
        errors = []
        for it in items:
            if not it.is_valid():
                errors.append(f"Invalid record: {it.image_path} (Empty text or missing path)")
                continue
            if not os.path.exists(it.image_path):
                errors.append(f"File missing on disk: {it.image_path}")
                continue
            valid.append(it)
        return valid, errors

    def build_splits(
        self,
        items: List[OCRItem],
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
        real_world_items: Optional[List[OCRItem]] = None,
        group_key: Optional[str] = "source_document_id"
    ) -> DatasetSplit:
        """Constructs leak-free train/val/test splits grouped by source document/publisher."""
        splits = BaseOCRDatasetLoader.create_leak_free_splits(
            items, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, group_key=group_key
        )
        if real_world_items:
            splits.real_world_test = real_world_items
        return splits

    def build_grouped_splits(
        self,
        items: List[OCRItem],
        group_key: str = "source_document_id",
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10
    ) -> DatasetSplit:
        """Explicit convenience alias for document/publisher grouped splits."""
        return self.build_splits(
            items, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, group_key=group_key
        )

    def export_split_manifest(self, splits: DatasetSplit, output_json_path: str):
        """Exports dataset split metadata for full reproducibility."""
        data = {
            "summary": splits.summary(),
            "train": [{"path": i.image_path, "text": i.transcription, "source": i.dataset_source, "doc_id": i.source_document_id} for i in splits.train],
            "val": [{"path": i.image_path, "text": i.transcription, "source": i.dataset_source, "doc_id": i.source_document_id} for i in splits.val],
            "test": [{"path": i.image_path, "text": i.transcription, "source": i.dataset_source, "doc_id": i.source_document_id} for i in splits.test],
            "real_world_test": [{"path": i.image_path, "text": i.transcription, "source": i.dataset_source, "doc_id": i.source_document_id} for i in splits.real_world_test]
        }
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Exported dataset manifest to {output_json_path} ({splits.total_samples} total samples)")
