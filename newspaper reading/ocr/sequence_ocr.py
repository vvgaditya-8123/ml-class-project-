import os
import sys
import cv2
import numpy as np
from typing import List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from schema import OCRResult, OCRStatus
from .base_recognizer import TextRecognizer
from .legacy_ocr_engine import CharCNN, OCREngine as LegacyCharOCREngine

class SequenceTextRecognizer(TextRecognizer):
    """
    Sequence-oriented Text Recognizer implementing the standardized TextRecognizer interface.
    Operates on full text-line crops instead of isolated single characters.
    Handles multi-line blocks via horizontal projection profiling.
    """

    def __init__(self, model_name: str = "SequenceOCR-Hybrid"):
        super().__init__(model_name=model_name)
        # Initialize legacy engine as a local character/word decoding fallback
        self._legacy_engine = None
        self._init_backend()

    def _init_backend(self):
        try:
            self._legacy_engine = LegacyCharOCREngine()
        except Exception as e:
            print(f"Notice: Legacy OCR Engine backend could not be initialized: {e}")
            self._legacy_engine = None

    def recognize_line(self, line_img: np.ndarray) -> OCRResult:
        """
        Recognizes text from a single horizontal line crop.
        Returns explicit status ('ok', 'empty', or 'error') with score.
        """
        if line_img is None or line_img.size == 0:
            return OCRResult(text="", status="empty", model_name=self.model_name, score=0.0)

        h, w = line_img.shape[:2]
        if h < 6 or w < 6:
            return OCRResult(text="", status="empty", model_name=self.model_name, score=0.0)

        # Check for blank / uniform crops
        gray = line_img if len(line_img.shape) == 2 else cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
        if np.std(gray) < 5.0:  # Flat background / blank scan margin
            return OCRResult(text="", status="empty", model_name=self.model_name, score=0.0)

        try:
            # Decode text line
            if self._legacy_engine is not None:
                text = self._legacy_engine.read_line(line_img).strip()
                if text:
                    # Clean repeated whitespace
                    clean_text = " ".join(text.split())
                    return OCRResult(
                        text=clean_text,
                        status="ok",
                        model_name=self.model_name,
                        score=0.85
                    )
            return OCRResult(text="", status="empty", model_name=self.model_name, score=0.0)
        except Exception as e:
            return OCRResult(text="", status="error", model_name=self.model_name, error_message=str(e), score=0.0)

    def segment_block_lines(self, block_img: np.ndarray) -> List[Tuple[int, int]]:
        """
        Segments a multi-line text block into horizontal line (y1, y2) bounds
        using horizontal projection profiles.
        """
        if block_img is None or block_img.size == 0:
            return []

        gray = block_img if len(block_img.shape) == 2 else cv2.cvtColor(block_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if h < 10 or w < 10:
            return [(0, h)]

        # Adaptive thresholding
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Horizontal projection
        proj = np.sum(thresh, axis=1) / 255.0
        line_indices = np.where(proj > 1.0)[0]

        if len(line_indices) == 0:
            return [(0, h)]

        lines_ranges = []
        in_line = False
        start = 0

        for r in range(h):
            has_text = proj[r] > 1.2
            if has_text and not in_line:
                in_line = True
                start = max(0, r - 2)
            elif not has_text and in_line:
                in_line = False
                end = min(h, r + 2)
                if end - start >= 6:
                    lines_ranges.append((start, end))

        if in_line and (h - start >= 6):
            lines_ranges.append((start, h))

        return lines_ranges if lines_ranges else [(0, h)]

    def recognize_block(self, block_img: np.ndarray) -> OCRResult:
        """
        Recognizes a multi-line text block (headline or body paragraph).
        Splits into text lines, recognizes each line, and aggregates results.
        """
        if block_img is None or block_img.size == 0:
            return OCRResult(text="", status="empty", model_name=self.model_name, score=0.0)

        lines_ranges = self.segment_block_lines(block_img)
        recognized_lines = []
        scores = []

        for y1, y2 in lines_ranges:
            line_crop = block_img[y1:y2, :]
            res = self.recognize_line(line_crop)
            if res.status == "ok" and res.text:
                recognized_lines.append(res.text)
                if res.score is not None:
                    scores.append(res.score)

        if not recognized_lines:
            return OCRResult(text="", status="empty", model_name=self.model_name, score=0.0)

        combined_text = "\n".join(recognized_lines)
        avg_score = float(np.mean(scores)) if scores else None
        return OCRResult(
            text=combined_text,
            status="ok",
            model_name=self.model_name,
            score=avg_score
        )
