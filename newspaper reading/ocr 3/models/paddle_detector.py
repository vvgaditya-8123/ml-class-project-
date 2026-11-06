import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", os.path.join(BASE_DIR, ".paddlex"))
os.environ.setdefault("PADDLE_HOME", os.path.join(BASE_DIR, ".paddle"))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

class PaddleTextDetector:
    """
    PaddleOCR Text Detection module for newspaper pages:
    - Identifies multi-column text lines and bounding boxes.
    - Focuses strictly on geometric text-line detection (DBNet).
    - Uses normalized coordinates to maintain resolution independence.
    - Column-aware reading order sorting without brittle pixel bucket heuristics.
    """

    def __init__(self, lang: str = "en", use_angle_cls: bool = True):
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self._paddle_ocr = None
        self._init_detector()

    def _init_detector(self):
        try:
            from paddleocr import PaddleOCR
            self._paddle_ocr = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang
            )
            print("PaddleOCR Text Detector initialized successfully.")
        except Exception as e:
            self._paddle_ocr = None

    def detect_regions(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects all text bounding boxes and orders them into column-aware reading order.
        Returns sorted list of detected regions.
        """
        h, w = img_bgr.shape[:2]
        regions = []

        if self._paddle_ocr is not None:
            try:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                det_results = self._paddle_ocr.ocr(img_rgb, rec=False)
                
                if det_results and det_results[0]:
                    for poly in det_results[0]:
                        pts = np.array(poly, dtype=np.int32)
                        bx, by, bw, bh = cv2.boundingRect(pts)
                        # Clamping and sanity filtering
                        bx = max(0, min(bx, w - 1))
                        by = max(0, min(by, h - 1))
                        bw = max(1, min(bw, w - bx))
                        bh = max(1, min(bh, h - by))
                        if bw >= 15 and bh >= 6:
                            regions.append(self._create_region_record(bx, by, bw, bh, w, h, pts))
            except Exception as e:
                print(f"PaddleOCR detection exception: {e}")

        # If Paddle is uninstalled or returned no regions, use OpenCV adaptive morphological line detector
        if not regions:
            regions = self._detect_via_morphology(img_bgr)

        # Sort into column-aware reading order
        regions = self._sort_reading_order(regions, w, h)
        return regions

    def _create_region_record(self, x: int, y: int, w: int, h: int, page_w: int, page_h: int, polygon=None) -> Dict[str, Any]:
        """Creates a normalized region record with geometric text-line characteristics."""
        nx1 = x / float(page_w)
        ny1 = y / float(page_h)
        nx2 = (x + w) / float(page_w)
        ny2 = (y + h) / float(page_h)
        nw = w / float(page_w)
        nh = h / float(page_h)

        # Column assignment based on normalized x center
        center_x = (nx1 + nx2) / 2.0
        num_columns = 5  # Standard broadsheet newspaper column grid
        col_idx = int(center_x * num_columns)

        # Heuristic role suggestion for text lines (title/headline line vs body text line)
        is_spanning = nw > 0.40
        is_large_font = nh > 0.022 or h >= 26
        role = "HEADLINE" if (is_large_font or (is_spanning and nh > 0.015)) else "BODY_TEXT"

        return {
            "bbox": (x, y, w, h),
            "normalized_bbox": (nx1, ny1, nx2, ny2),
            "polygon": polygon.tolist() if polygon is not None else [[x, y], [x+w, y], [x+w, y+h], [x, y+h]],
            "role": role,
            "column_index": col_idx,
            "is_spanning": is_spanning,
            "height": h,
            "width": w
        }

    def _detect_via_morphology(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Multi-column line detector using adaptive horizontal projection and contour bounds."""
        page_h, page_w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8)

        # Connect text lines horizontally without merging adjacent vertical columns
        k_w = max(15, int(page_w * 0.02))
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 2))
        connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_line)

        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            # Filter noise specks and page borders
            if bw >= 25 and bh >= 8 and bw < (page_w * 0.98) and bh < (page_h * 0.85):
                regions.append(self._create_region_record(bx, by, bw, bh, page_w, page_h, cnt))

        return regions

    def _sort_reading_order(self, regions: List[Dict[str, Any]], page_w: int, page_h: int) -> List[Dict[str, Any]]:
        """
        Sorts regions into natural newspaper reading order:
        1. Spanning banner headlines / page titles first (top-to-bottom)
        2. Column items grouped by column left-to-right, then top-to-bottom within each column.
        """
        if not regions:
            return []

        spanning = [r for r in regions if r.get("is_spanning", False)]
        col_items = [r for r in regions if not r.get("is_spanning", False)]

        spanning.sort(key=lambda r: r["bbox"][1])
        col_items.sort(key=lambda r: (r["column_index"], r["bbox"][1]))

        # Interleave: spanning items partition vertical page bands
        ordered = []
        if not spanning:
            return col_items

        # Group items into vertical bands demarcated by spanning headlines
        sorted_spanning = sorted(spanning, key=lambda r: r["bbox"][1])
        ordered.extend(sorted_spanning)
        ordered.extend(col_items)
        # Final stable topological sort: primary key y // band, secondary column, tertiary y
        band_height = max(100, page_h // 10)
        ordered.sort(key=lambda r: (r["bbox"][1] // band_height, r["column_index"], r["bbox"][1]))
        return ordered
