import os
import sys
import json
import cv2
import numpy as np
import joblib

# Ensure stdout uses UTF-8
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "layout_model.joblib")
META_PATH = os.path.join(BASE_DIR, "layout_metadata.json")

CLASS_HEADLINE = 0
CLASS_BODY = 1
CLASS_BANNER = 2

class LayoutEngine:
    def __init__(self, model_path=MODEL_PATH, meta_path=META_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Layout model not found at {model_path}. Please run train_layout.py first.")
        self.model = joblib.load(model_path)
        self.metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def extract_block_features(self, img_gray, x, y, w, h, page_w, page_h):
        rel_h = h / float(page_h)
        rel_w = w / float(page_w)
        aspect = w / float(max(1, h))
        rel_y = y / float(page_h)
        rel_x = x / float(page_w)
        rel_area = (w * h) / float(page_w * page_h)

        crop = img_gray[y:y+h, x:x+w]
        if crop.size == 0:
            return None

        thresh = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        ink_density = np.sum(thresh == 255) / float(w * h)

        proj = np.sum(thresh == 255, axis=1)
        text_rows = np.where(proj > (w * 0.05))[0]
        line_count = 1
        if len(text_rows) > 0:
            diffs = np.diff(text_rows)
            gaps = np.sum(diffs > 3)
            line_count = max(1, gaps + 1)
        avg_line_height = float(h) / float(line_count)
        col_span = w / float(page_w * 0.22)

        return [
            rel_h, rel_w, aspect, rel_y, rel_x,
            rel_area, ink_density, avg_line_height, line_count, col_span
        ]

    def segment_layout(self, img_bgr: np.ndarray):
        """
        Extracts candidate blocks, predicts their layout class (Headline vs Body vs Banner),
        and organizes them into structured article units.
        """
        page_h, page_w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8)

        # Morphological horizontal line/block grouping
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 3))
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, h_kernel)

        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw < 45 or bh < 10:
                continue
            feats = self.extract_block_features(gray, bx, by, bw, bh, page_w, page_h)
            if feats is not None:
                candidates.append({
                    "box": (bx, by, bw, bh),
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
                if oy > hy: # below this headline
                    # Check horizontal overlap
                    overlap_x1 = max(hx, ox)
                    overlap_x2 = min(hl_right, ox + ow)
                    if overlap_x2 - overlap_x1 > 30: # overlapping column
                        if oy < lower_bound_y:
                            lower_bound_y = oy

            # Collect body text blocks falling within this article region
            matched_body = []
            for b_idx, b in enumerate(body_blocks):
                bx, by, bw, bh = b["box"]
                # Must be vertically below the headline and above the lower boundary
                if (by >= (hl_bottom - 10)) and (by + bh <= (lower_bound_y + 15)):
                    # Check horizontal alignment (within column span of the headline)
                    overlap_x1 = max(hx - 25, bx)
                    overlap_x2 = min(hl_right + 25, bx + bw)
                    if (overlap_x2 - overlap_x1) > (bw * 0.4):
                        matched_body.append(b)
                        assigned_body_indices.add(b_idx)

            # Sort matched body blocks in reading order (top to bottom)
            matched_body.sort(key=lambda b: (b["box"][1], b["box"][0]))

            articles.append({
                "article_id": idx + 1,
                "headline_box": (hx, hy, hw, hh),
                "headline_crop": img_bgr[hy:hy+hh, hx:hx+hw],
                "headline_conf": hl["conf"],
                "body_boxes": [b["box"] for b in matched_body],
                "body_crops": [img_bgr[b['box'][1]:b['box'][1]+b['box'][3], b['box'][0]:b['box'][0]+b['box'][2]] for b in matched_body]
            })

        return articles

if __name__ == "__main__":
    engine = LayoutEngine()
    news_sample = os.path.join(BASE_DIR, "News", "0000023720.jpg")
    if os.path.exists(news_sample):
        im = cv2.imread(news_sample)
        res = engine.segment_layout(im)
        print(f"Detected {len(res)} distinct articles on sample newspaper page:")
        for a in res:
            print(f"  Article {a['article_id']}: Headline box {a['headline_box']}, Body blocks count: {len(a['body_boxes'])}")

