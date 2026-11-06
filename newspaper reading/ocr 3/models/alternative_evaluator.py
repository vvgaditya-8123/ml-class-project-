import time
import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2

try:
    from evaluation.metrics import character_error_rate, word_error_rate, normalized_edit_distance
except (ImportError, ValueError):
    try:
        from ..evaluation.metrics import character_error_rate, word_error_rate, normalized_edit_distance
    except (ImportError, ValueError):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from evaluation.metrics import character_error_rate, word_error_rate, normalized_edit_distance

class AlternativeModelEvaluator:
    """
    Experimental evaluation harness to benchmark alternative OCR engines
    on the exact same test dataset:
    - PaddleOCR (Production Default when installed)
    - EasyOCR (Alternative baseline)
    - Microsoft TrOCR (Transformer Vision-Encoder-Decoder via transformers)
    - DocTR (Document Text Recognition)
    """

    def __init__(self):
        self.available_engines = {}
        self._detect_engines()

    def _detect_engines(self):
        # 1. PaddleOCR
        try:
            import paddleocr
            self.available_engines["PaddleOCR"] = {"status": "available", "instance": None}
        except ImportError:
            self.available_engines["PaddleOCR"] = {"status": "not_installed"}

        # 2. EasyOCR
        try:
            import easyocr
            self.available_engines["EasyOCR"] = {"status": "available", "instance": None}
        except ImportError:
            self.available_engines["EasyOCR"] = {"status": "not_installed"}

        # 3. Microsoft TrOCR
        try:
            import transformers
            self.available_engines["Microsoft TrOCR"] = {"status": "available", "instance": None}
        except ImportError:
            self.available_engines["Microsoft TrOCR"] = {"status": "requires 'transformers'"}

        # 4. DocTR
        try:
            import doctr
            self.available_engines["DocTR"] = {"status": "available", "instance": None}
        except ImportError:
            self.available_engines["DocTR"] = {"status": "requires 'python-doctr'"}

    def get_easyocr_instance(self):
        if self.available_engines["EasyOCR"]["instance"] is None and self.available_engines["EasyOCR"]["status"] == "available":
            import easyocr
            self.available_engines["EasyOCR"]["instance"] = easyocr.Reader(['en'], gpu=False)
        return self.available_engines["EasyOCR"]["instance"]

    def run_inference(self, engine_name: str, img_bgr: np.ndarray) -> Tuple[str, float, float, str]:
        """Runs single engine inference, returning (predicted_text, confidence, latency_ms, status)."""
        t0 = time.perf_counter()

        if engine_name == "EasyOCR":
            inst = self.get_easyocr_instance()
            if inst:
                results = inst.readtext(img_bgr)
                texts = [r[1] for r in results]
                confs = [r[2] for r in results]
                lat = (time.perf_counter() - t0) * 1000
                pred = " ".join(texts)
                return pred, float(np.mean(confs)) if confs else 0.0, lat, "ok" if pred else "empty"

        elif engine_name == "PaddleOCR":
            try:
                from paddleocr import PaddleOCR
                if self.available_engines["PaddleOCR"]["instance"] is None:
                    self.available_engines["PaddleOCR"]["instance"] = PaddleOCR(use_angle_cls=False, lang="en")
                inst = self.available_engines["PaddleOCR"]["instance"]
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                res = inst.ocr(img_rgb, det=True, rec=True)
                texts, confs = [], []
                if res and res[0]:
                    for line in res[0]:
                        texts.append(line[1][0])
                        confs.append(line[1][1])
                lat = (time.perf_counter() - t0) * 1000
                pred = " ".join(texts)
                return pred, float(np.mean(confs)) if confs else 0.0, lat, "ok" if pred else "empty"
            except Exception as e:
                lat = (time.perf_counter() - t0) * 1000
                return "", 0.0, lat, f"error: {str(e)}"

        # Default for unsupported or uninstalled engines: honest empty result with explicit status
        lat = (time.perf_counter() - t0) * 1000
        engine_status = self.available_engines.get(engine_name, {}).get("status", "unavailable")
        return "", 0.0, lat, engine_status

    def benchmark_sample(self, img_bgr: np.ndarray, ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """Runs available engines on the same test sample and computes actual error metrics when ground truth is provided."""
        report = {}
        for eng in ["PaddleOCR", "EasyOCR", "Microsoft TrOCR", "DocTR"]:
            pred, conf, lat, status = self.run_inference(eng, img_bgr)
            entry = {
                "prediction": pred,
                "confidence": conf,
                "latency_ms": round(lat, 2),
                "status": status
            }
            if ground_truth is not None and pred:
                entry["cer"] = round(character_error_rate(ground_truth, pred), 4)
                entry["wer"] = round(word_error_rate(ground_truth, pred), 4)
                entry["ned"] = round(normalized_edit_distance(ground_truth, pred), 4)
            report[eng] = entry
        return report
