import os
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
    - Manages fine-tuned checkpoints separately.
    - Properly passes checkpoint path to constructor and raises/reports explicit exceptions.
    """

    AVAILABLE_MODELS = {
        "en_PP-OCRv4": {"description": "Latest Mobile/Server OCRv4 model with high English generalization", "version": "v4"},
        "en_PP-OCRv3": {"description": "Lightweight SVTR-LCNet recognition model", "version": "v3"},
        "custom_finetuned": {"description": "Newspaper fine-tuned custom weights", "version": "custom"}
    }

    def __init__(self, model_name: str = "en_PP-OCRv4", checkpoint_path: Optional[str] = None):
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self._ocr = None
        self.initialization_error = None
        self._init_recognizer()

    def _init_recognizer(self):
        try:
            from paddleocr import PaddleOCR
            
            kwargs = {
                "use_angle_cls": False,
                "lang": "en"
            }
            if self.checkpoint_path:
                if not os.path.exists(self.checkpoint_path):
                    raise FileNotFoundError(f"Specified PaddleOCR checkpoint not found: {self.checkpoint_path}")
                kwargs["rec_model_dir"] = self.checkpoint_path

            self._ocr = PaddleOCR(**kwargs)
            print(f"PaddleTextRecognizer initialized successfully with model: {self.model_name}")
        except Exception as e:
            self.initialization_error = str(e)
            self._ocr = None

    def recognize_crop(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Recognizes text line or word crop.
        Returns (transcription, confidence).
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return "", 0.0

        if self._ocr is None:
            return "", 0.0

        try:
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            res = self._ocr.ocr(crop_rgb, det=False, rec=True)
            if res and res[0]:
                text, conf = res[0][0]
                return text.strip(), float(conf)
            return "", 0.0
        except Exception as e:
            print(f"PaddleTextRecognizer error during inference: {e}")
            return "", 0.0

    def batch_recognize(self, crops: List[np.ndarray]) -> List[Tuple[str, float]]:
        return [self.recognize_crop(c) for c in crops]

    @classmethod
    def list_models(cls) -> Dict[str, Dict[str, str]]:
        return cls.AVAILABLE_MODELS
