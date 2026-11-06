import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from schema import OCRResult, OCRStatus
from .base_recognizer import TextRecognizer
from .sequence_ocr import SequenceTextRecognizer
from .legacy_ocr_engine import CharCNN, OCREngine as LegacyCharOCREngine

class OCREngine:
    """
    Standardized OCR Engine wrapper:
    - Delegates text recognition to SequenceTextRecognizer.
    - Exposes recognize() returning standardized OCRResult.
    - Provides backward-compatible read_text_block() and read_line() returning clean string.
    - Preserves predict_char() for legacy character benchmarking.
    """

    def __init__(self, model_name: str = "SequenceOCR"):
        self.model_name = model_name
        self.recognizer = SequenceTextRecognizer(model_name=model_name)
        self.legacy_char_engine = LegacyCharOCREngine()

    def recognize(self, img: np.ndarray) -> OCRResult:
        """Primary API: Recognizes text from a line or block crop, returning OCRResult."""
        return self.recognizer.recognize_block(img)

    def read_text_block(self, block_img: np.ndarray) -> str:
        """Backward-compatible helper returning recognized text string."""
        res = self.recognize(block_img)
        return res.text

    def read_line(self, line_img: np.ndarray) -> str:
        """Backward-compatible helper returning recognized text for a single line."""
        res = self.recognizer.recognize_line(line_img)
        return res.text

    def predict_char(self, char_img: np.ndarray):
        """Preserved for isolated character evaluation on synthetic benchmarks."""
        return self.legacy_char_engine.predict_char(char_img)
