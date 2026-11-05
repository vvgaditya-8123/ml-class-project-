import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

# Ensure Paddle environment variables are bound to local workspace
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", os.path.join(BASE_DIR, ".paddlex"))
os.environ.setdefault("PADDLE_HOME", os.path.join(BASE_DIR, ".paddle"))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

class PaddleTextDetector:
    """
    PaddleOCR Text Detection module for newspaper pages:
    - Identifies multi-column text lines and bounding boxes.
    - Detects prominent headlines vs paragraph regions.
    - Handles text rotation / orientations.
    """

    def __init__(self, lang: str = "en", use_angle_cls: bool = True):
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self._paddle_ocr = None
        self._init_detector()

    def _init_detector(self):
        try:
            from paddleocr import PaddleOCR
            # Initialize with detection enabled, recognition deferred
            self._paddle_ocr = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang
            )
            print("PaddleOCR Text Detector initialized successfully.")
        except Exception as e:
            print(f"PaddleOCR detection fallback to high-resolution OpenCV DBNet/morphology: {e}")
            self._paddle_ocr = None

    def detect_regions(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects all text bounding boxes, classifies layout roles (Headline vs Body),
        and handles multi-column ordering.
        Returns sorted list of detected regions.
        """
        h, w = img_bgr.shape[:2]
        regions = []

        if self._paddle_ocr is not None:
            try:
                # PaddleOCR expects RGB or image path
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                det_results = self._paddle_ocr.ocr(img_rgb, rec=False)
                
                # Format: list of 4-point polygons [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                if det_results and det_results[0]:
                    for poly in det_results[0]:
                        pts = np.array(poly, dtype=np.int32)
                        bx, by, bw, bh = cv2.boundingRect(pts)
                        if bw >= 20 and bh >= 8:
                            regions.append(self._classify_region(bx, by, bw, bh, w, h, pts))
            except Exception as e:
                print(f"PaddleOCR detection invocation error: {e}")

        # If regions empty or fallback active, run advanced multi-column newspaper detector
        if not regions:
            regions = self._detect_via_morphology(img_bgr)

        # Sort reading order (multi-column: group columns then top-to-bottom)
        regions = self._sort_reading_order(regions, w, h)
        return regions

    def _classify_region(self, x: int, y: int, w: int, h: int, page_w: int, page_h: int, polygon=None) -> Dict[str, Any]:
        """Classifies text box into HEADLINE vs COLUMN_BODY vs BANNER."""
        rel_h = h / float(page_h)
        rel_w = w / float(page_w)

        # Headlines have prominent font height or span across multiple columns
        is_headline = (h >= 30) or (rel_w > 0.40 and h >= 22) or (h >= page_h * 0.025)
        is_banner = (y < page_h * 0.12) and (rel_w > 0.65 or h > 100)

        role = "BANNER" if is_banner else ("HEADLINE" if is_headline else "BODY_TEXT")
        col_idx = int(x / (page_w * 0.22)) # Estimated newspaper column index

        return {
            "bbox": (x, y, w, h),
            "polygon": polygon.tolist() if polygon is not None else [[x, y], [x+w, y], [x+w, y+h], [x, y+h]],
            "role": role,
            "column_index": col_idx,
            "height": h,
            "width": w
        }

    def _detect_via_morphology(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """High-resolution multi-column text detector using adaptive morphological projection."""
        page_h, page_w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8)

        # Horizontal grouping for lines
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_line)

        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw >= 35 and bh >= 10:
                regions.append(self._classify_region(bx, by, bw, bh, page_w, page_h, cnt))

        return regions

    def _sort_reading_order(self, regions: List[Dict[str, Any]], page_w: int, page_h: int) -> List[Dict[str, Any]]:
        """Sorts regions into proper newspaper reading order (top-to-bottom within column bands)."""
        # Separate wide headlines/banners from standard column bodies
        wide_items = [r for r in regions if r["role"] in ("HEADLINE", "BANNER") and r["width"] > (page_w * 0.40)]
        col_items = [r for r in regions if r not in wide_items]

        # Sort wide items by y
        wide_items.sort(key=lambda r: r["bbox"][1])

        # Sort column items by column then y
        col_items.sort(key=lambda r: (r["column_index"], r["bbox"][1]))

        # Interleave in natural top-down reading order
        combined = sorted(regions, key=lambda r: (r["bbox"][1] // 150, r["column_index"], r["bbox"][1]))
        return combined
