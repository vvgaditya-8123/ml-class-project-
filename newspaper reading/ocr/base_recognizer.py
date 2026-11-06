import os
import sys
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

# Ensure schema is accessible
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from schema import OCRResult, OCRStatus

class TextRecognizer(ABC):
    """
    Standard abstract interface for all newspaper text recognizers.
    Decouples line detection and page layout from specific OCR model architectures.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def recognize_line(self, line_image: np.ndarray) -> OCRResult:
        """
        Recognizes a single horizontal line crop.
        Returns standardized OCRResult with true text, optional confidence score, and status.
        """
        raise NotImplementedError

    def recognize_lines(self, line_images: List[np.ndarray]) -> List[OCRResult]:
        """Batch recognition over multiple text line crops."""
        return [self.recognize_line(img) for img in line_images]
