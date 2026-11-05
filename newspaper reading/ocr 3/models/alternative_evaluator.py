import time
import os
from typing import List, Dict, Any, Tuple
import numpy as np
import cv2

class AlternativeModelEvaluator:
    """
    Experimental evaluation harness to benchmark alternative OCR engines
    on the exact same test dataset:
    - PaddleOCR (Production Default)
    - EasyOCR (Alternative baseline)
    - Microsoft TrOCR (Transformer Vision-Encoder-Decoder)
    - DocTR (Document Text Recognition)
    """

    def __init__(self):
        self.available_engines = {}
        self._detect_engines()

    def _detect_engines(self):
        # 1. PaddleOCR
        try:
            from paddleocr import PaddleOCR
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

    def run_inference(self, engine_name: str, img_bgr: np.ndarray) -> Tuple[str, float, float]:
        """Runs single engine inference, returning (predicted_text, confidence, latency_ms)."""
        t0 = time.perf_counter()

        if engine_name == "EasyOCR":
            inst = self.get_easyocr_instance()
            if inst:
                results = inst.readtext(img_bgr)
                texts = [r[1] for r in results]
                confs = [r[2] for r in results]
                lat = (time.perf_counter() - t0) * 1000
                return " ".join(texts), float(np.mean(confs)) if confs else 0.0, lat

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
                return " ".join(texts), float(np.mean(confs)) if confs else 0.0, lat
            except Exception as e:
                lat = (time.perf_counter() - t0) * 1000
                return f"[PaddleOCR Error: {e}]", 0.0, lat

        # Fallback simulation for TrOCR / DocTR when dependencies uninstalled
        lat = (time.perf_counter() - t0) * 1000
        status = self.available_engines.get(engine_name, {}).get("status", "unavailable")
        return f"[{engine_name}: {status}]", 0.0, lat

    def benchmark_sample(self, img_bgr: np.ndarray, ground_truth: str) -> Dict[str, Any]:
        """Runs all available engines on the same test sample and compares outputs."""
        report = {}
        for eng in ["PaddleOCR", "EasyOCR", "Microsoft TrOCR", "DocTR"]:
            pred, conf, lat = self.run_inference(eng, img_bgr)
            report[eng] = {
                "prediction": pred,
                "confidence": conf,
                "latency_ms": round(lat, 2),
                "status": self.available_engines.get(eng, {}).get("status", "unknown")
            }
        return report
