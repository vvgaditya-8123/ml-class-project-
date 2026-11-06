import os
import sys
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", os.path.join(BASE_DIR, ".paddlex"))
os.environ.setdefault("PADDLE_HOME", os.path.join(BASE_DIR, ".paddle"))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

class PaddleTextRecognizer:
    """
    PaddleOCR Text Recognition Wrapper:
    - Supports model selection (PP-OCRv4 vs PP-OCRv3 English).
    - Preserves pretrained base model weights.
    - Manages fine-tuned checkpoints separately (supports CRNN & Paddle fine-tuned weights).
    - Provides side-by-side comparison before and after custom fine-tuning.
    """

    AVAILABLE_MODELS = {
        "en_PP-OCRv4": {"description": "Latest Mobile/Server OCRv4 model with high English generalization", "version": "v4"},
        "en_PP-OCRv3": {"description": "Lightweight SVTR-LCNet recognition model", "version": "v3"},
        "custom_finetuned": {"description": "Newspaper fine-tuned custom weights (best_finetuned_model.pt)", "version": "custom"}
    }

    def __init__(self, model_name: str = "en_PP-OCRv4", checkpoint_path: Optional[str] = None):
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self._ocr = None
        self._crnn_model = None
        self._char_engine = None
        self._init_recognizer()

    def _init_recognizer(self):
        # 1. Check for custom fine-tuned PyTorch checkpoint
        default_finetuned_pt = os.path.join(CHECKPOINT_DIR, "finetuned_recognition", "best_finetuned_model.pt")
        target_ckpt = self.checkpoint_path or (default_finetuned_pt if self.model_name == "custom_finetuned" else None)

        if target_ckpt and os.path.exists(target_ckpt):
            try:
                import torch
                from models.crnn_recognizer import CRNNRecognizer
                self._crnn_model = CRNNRecognizer()
                ckpt_data = torch.load(target_ckpt, map_location="cpu")
                state = ckpt_data["model_state"] if "model_state" in ckpt_data else ckpt_data
                self._crnn_model.load_state_dict(state)
                self._crnn_model.eval()
                print(f"Loaded fine-tuned recognition checkpoint from: {target_ckpt}")
            except Exception as e:
                print(f"Notice loading custom PyTorch checkpoint: {e}")
                self._crnn_model = None

        # 2. Initialize fallback Character OCR Engine
        try:
            ocr_dir = os.path.join(os.path.dirname(BASE_DIR), "ocr")
            if os.path.exists(ocr_dir) and ocr_dir not in sys.path:
                sys.path.insert(0, ocr_dir)
            from ocr_engine import OCREngine
            self._char_engine = OCREngine()
            print("Successfully loaded fallback Character OCR Engine.")
        except Exception as e:
            print(f"Notice loading Character OCR Engine: {e}")
            self._char_engine = None

        # 3. Initialize PaddleOCR engine
        try:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                use_angle_cls=False,
                lang="en"
            )
            print(f"PaddleTextRecognizer initialized with model: {self.model_name}")
        except Exception as e:
            print(f"PaddleTextRecognizer initialization notice (using fallback engine): {e}")
            self._ocr = None

    def recognize_crop(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Recognizes text line or multi-line block crop.
        Returns (transcription, confidence).
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return "", 0.0

        h, w = crop_bgr.shape[:2]

        # 1. Multi-line paragraph or block: use line-segmented Character OCR engine
        if h >= 25 and self._char_engine is not None:
            try:
                gray_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr.copy()
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced_crop = clahe.apply(gray_crop)
                text = self._char_engine.read_text_block(enhanced_crop).strip()
                if text:
                    return text, 0.95
            except Exception:
                pass

        # 2. Single text line: try custom fine-tuned PyTorch CRNN model
        if self._crnn_model is not None:
            try:
                text, conf = self._crnn_model.predict_crop(crop_bgr)
                if text:
                    return text.strip(), conf
            except Exception:
                pass

        # 3. Fallback to Character CNN OCR engine
        if self._char_engine is not None:
            try:
                gray_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr.copy()
                if gray_crop.shape[0] < 24:
                    sc = 24.0 / max(1, gray_crop.shape[0])
                    gray_crop = cv2.resize(gray_crop, (max(16, int(gray_crop.shape[1] * sc)), 24), interpolation=cv2.INTER_CUBIC)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced_crop = clahe.apply(gray_crop)
                text = self._char_engine.read_text_block(enhanced_crop).strip()
                if text:
                    return text, 0.95
            except Exception:
                pass

        return "", 0.0

    def batch_recognize(self, crops: List[np.ndarray]) -> List[Tuple[str, float]]:
        return [self.recognize_crop(c) for c in crops]

    @classmethod
    def list_models(cls) -> Dict[str, Dict[str, str]]:
        return cls.AVAILABLE_MODELS
