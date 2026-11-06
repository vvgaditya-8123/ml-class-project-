import os
import sys
import json
import joblib
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "layout_model.joblib")
META_PATH = os.path.join(BASE_DIR, "layout_metadata.json")

CLASS_BODY = 0
CLASS_HEADLINE = 1
CLASS_BANNER = 2
CLASS_NAMES = {0: "BODY_TEXT", 1: "HEADLINE", 2: "BANNER"}

class LayoutEngine:
    """
    Newspaper Layout Segmentation Engine:
    - Extracts layout candidate regions with normalized coordinates.
    - Classifies candidate regions into semantic roles (Headline, Body, Banner).
    - Groups headlines and body columns into cohesive article units.
    - Enforces unique body block ownership to prevent duplicate assignment across articles.
    """

    def __init__(self, model_path=MODEL_PATH, meta_path=META_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained layout model not found at: {model_path}")

        self.model = joblib.load(model_path)
        self.metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def extract_block_features(self, gray: np.ndarray, x: int, y: int, w: int, h: int, page_w: int, page_h: int):
        """
        Extracts 10 layout geometry & texture features:
        [rel_h, rel_w, aspect, rel_y, rel_x, rel_area, ink_density, avg_line_height, line_count, col_span]
        """
        crop = gray[y:y+h, x:x+w]
        if crop.size == 0:
            return None

        rel_h = float(h) / float(page_h)
        rel_w = float(w) / float(page_w)
        aspect = float(w) / float(max(1, h))
        rel_y = float(y) / float(page_h)
        rel_x = float(x) / float(page_w)
        rel_area = float(w * h) / float(page_w * page_h)

        # Ink density (foreground pixel ratio)
        thresh = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        ink_density = float(np.count_nonzero(thresh)) / float(max(1, w * h))

        # Horizontal projection to estimate text lines inside block
        h_proj = np.sum(thresh, axis=1) / 255.0
        gaps = 0
        in_gap = False
        for val in h_proj:
            if val < 2.0 and not in_gap:
                in_gap = True
                gaps += 1
            elif val >= 2.0 and in_gap:
                in_gap = False

        line_count = max(1, gaps + 1)
        avg_line_height = float(h) / float(line_count)
        col_span = w / float(max(1, page_w * 0.22))

        return [
            rel_h, rel_w, aspect, rel_y, rel_x,
            rel_area, ink_density, avg_line_height, line_count, col_span
        ]

    def segment_layout(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extracts candidate blocks, predicts their layout class (Headline vs Body vs Banner),
        and organizes them into structured article units with unique body assignments.
        """
        page_h, page_w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8)

        # Morphological horizontal line/block grouping with responsive kernel
        kernel_w = max(15, min(35, int(page_w * 0.02)))
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, h_kernel)

        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            # Boundary sanity checks: filter out whole-page borders and specks
            if bw < 35 or bh < 8:
                continue
            if bw > (page_w * 0.98) and bh > (page_h * 0.85):
                continue

            feats = self.extract_block_features(gray, bx, by, bw, bh, page_w, page_h)
            if feats is not None:
                candidates.append({
                    "box": (bx, by, bw, bh),
                    "normalized_box": (bx / page_w, by / page_h, (bx + bw) / page_w, (by + bh) / page_h),
                    "feats": feats
                })

        if not candidates:
            return []

        # Predict layout classes using trained model
        X = [c["feats"] for c in candidates]
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)

        headlines = []
        body_blocks = []
        banners = []

        for i, c in enumerate(candidates):
            c["label"] = int(preds[i])
            c["conf"] = float(np.max(probs[i]))

            if c["label"] == CLASS_HEADLINE:
                headlines.append(c)
            elif c["label"] == CLASS_BODY:
                body_blocks.append(c)
            else:
                banners.append(c)

        # Sort headlines in reading order (top-to-bottom, left-to-right)
        headlines.sort(key=lambda h: (h["box"][1], h["box"][0]))

        # Group body text blocks under their corresponding headlines
        articles = []
        assigned_body_indices = set()

        for idx, hl in enumerate(headlines):
            hx, hy, hw, hh = hl["box"]
            hl_bottom = hy + hh
            hl_right = hx + hw

            # Determine the lower vertical boundary (next headline below in overlapping columns)
            lower_bound_y = page_h
            for other_hl in headlines:
                ox, oy, ow, oh = other_hl["box"]
                if oy > hy:  # strictly below this headline
                    overlap_x1 = max(hx, ox)
                    overlap_x2 = min(hl_right, ox + ow)
                    if (overlap_x2 - overlap_x1) > 25:
                        if oy < lower_bound_y:
                            lower_bound_y = oy

            # Collect body text blocks with unique ownership check
            matched_body = []
            for b_idx, b in enumerate(body_blocks):
                if b_idx in assigned_body_indices:
                    continue

                bx, by, bw, bh = b["box"]
                # Must be vertically below the headline and above the lower boundary
                if (by >= (hl_bottom - 10)) and (by + bh <= (lower_bound_y + 15)):
                    # Check horizontal alignment (within column span of the headline)
                    overlap_x1 = max(hx - 25, bx)
                    overlap_x2 = min(hl_right + 25, bx + bw)
                    if (overlap_x2 - overlap_x1) > (bw * 0.35):
                        matched_body.append(b)
                        assigned_body_indices.add(b_idx)

            # Sort matched body blocks in reading order (top to bottom)
            matched_body.sort(key=lambda b: (b["box"][1], b["box"][0]))

            # Clamp crops safely
            hl_crop = img_bgr[max(0, hy):min(page_h, hy+hh), max(0, hx):min(page_w, hx+hw)]
            body_crops = [
                img_bgr[max(0, b['box'][1]):min(page_h, b['box'][1]+b['box'][3]),
                        max(0, b['box'][0]):min(page_w, b['box'][0]+b['box'][2])]
                for b in matched_body
            ]

            articles.append({
                "article_id": idx + 1,
                "headline_box": (hx, hy, hw, hh),
                "normalized_headline_box": hl["normalized_box"],
                "headline_crop": hl_crop,
                "headline_conf": hl["conf"],
                "layout_score": hl["conf"],
                "body_boxes": [b["box"] for b in matched_body],
                "normalized_body_boxes": [b["normalized_box"] for b in matched_body],
                "body_crops": body_crops
            })

        return articles
