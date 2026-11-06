import os
import sys
import cv2
import numpy as np
from typing import Dict, Any, List, Optional

# Ensure local imports work cleanly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from schema import OCRResult, ArticleResult, LayoutRegion, OCRStatus
from layout.layout_engine import LayoutEngine
from ocr.ocr_engine import OCREngine

class NewspaperReader:
    """
    Unified Newspaper Reading Pipeline:
    - Segments newspaper page layout into articles and columns (LayoutEngine).
    - Recognizes text with sequence-aware OCR (OCREngine).
    - Maintains strict output integrity: NEVER fabricates fallback text or placeholder stories.
    - Exposes separate layout detection and OCR recognition scores.
    """

    def __init__(self):
        print("Initializing NewspaperReader (Layout Engine + OCR Engine)...")
        self.layout_engine = LayoutEngine()
        self.ocr_engine = OCREngine()
        print("NewspaperReader initialized successfully.")

    def enhance_crop_for_ocr(self, crop: np.ndarray) -> np.ndarray:
        """
        Enhances text contrast and sharpness for optimal OCR reading.
        Clamps dimensions and rescales small fonts cleanly.
        """
        if crop is None or crop.size == 0:
            return crop

        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        h, w = gray.shape
        if h < 24:
            scale = 24.0 / max(1, h)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced

    def process_newspaper(self, image_source) -> Dict[str, Any]:
        """
        Takes image path or cv2 image matrix:
        1. Runs layout segmentation model to detect and isolate articles & headlines.
        2. Runs OCR on each headline without synthetic fallback.
        3. Runs OCR on corresponding body text beneath each headline without canned paragraphs.
        Returns full structured article data following the standardized schema.
        """
        if isinstance(image_source, str):
            if not os.path.exists(image_source):
                raise FileNotFoundError(f"Newspaper image not found: {image_source}")
            img_bgr = cv2.imread(image_source)
            image_path = image_source
        else:
            img_bgr = image_source
            image_path = "In-Memory Image"

        if img_bgr is None:
            raise ValueError("Failed to load valid newspaper image.")

        page_h, page_w = img_bgr.shape[:2]

        # 1. Layout Segmentation
        raw_articles = self.layout_engine.segment_layout(img_bgr)
        articles_data = []

        for idx, art in enumerate(raw_articles):
            a_id = idx + 1
            hl_box = art["headline_box"]
            hl_crop = art["headline_crop"]
            body_boxes = art["body_boxes"]
            body_crops = art["body_crops"]
            layout_conf = art.get("layout_score", art.get("headline_conf", 0.0))

            # 2. OCR on Headline
            enhanced_hl = self.enhance_crop_for_ocr(hl_crop)
            hl_res: OCRResult = self.ocr_engine.recognize(enhanced_hl)
            headline_text = " ".join(hl_res.text.split()) if hl_res.text else ""
            hl_status: OCRStatus = "ok" if headline_text else "empty"
            hl_ocr_score = hl_res.score if headline_text else None

            # 3. OCR on Body Text blocks under this headline
            body_paragraphs = []
            body_scores = []
            for b_crop in body_crops:
                enhanced_body = self.enhance_crop_for_ocr(b_crop)
                b_res: OCRResult = self.ocr_engine.recognize(enhanced_body)
                clean_body = b_res.text.strip()
                if clean_body:
                    body_paragraphs.append(clean_body)
                    if b_res.score is not None:
                        body_scores.append(b_res.score)

            combined_body = "\n\n".join(body_paragraphs).strip()
            body_status: OCRStatus = "ok" if combined_body else "empty"
            body_ocr_score = float(np.mean(body_scores)) if body_scores else None

            # Honest warnings: NEVER fabricate fake text
            warnings = []
            if not headline_text:
                warnings.append("Headline text could not be recognized.")
            if not combined_body:
                warnings.append("No body text was recognized in column blocks.")

            word_count = len(combined_body.split())

            article_res = ArticleResult(
                article_id=str(a_id),
                headline=headline_text,
                body_text=combined_body,
                headline_status=hl_status,
                body_status=body_status,
                layout_score=layout_conf,
                headline_ocr_score=hl_ocr_score,
                body_ocr_score=body_ocr_score,
                headline_box=hl_box,
                body_boxes=body_boxes,
                word_count=word_count,
                warnings=warnings,
                metadata={
                    "normalized_headline_box": art.get("normalized_headline_box"),
                    "normalized_body_boxes": art.get("normalized_body_boxes", [])
                }
            )

            articles_data.append(article_res.to_dict())

        return {
            "image_path": image_path,
            "dimensions": (page_w, page_h),
            "total_articles": len(articles_data),
            "articles": articles_data
        }

    def render_visual_overlay(self, img_bgr: np.ndarray, articles: List[Dict[str, Any]], selected_id: int = None) -> np.ndarray:
        """
        Draws visual layout bounding boxes on the newspaper image:
        - Highlights the selected headline and its associated body text prominently.
        - Dims non-selected articles.
        - Accurately renders boxes on original page coordinates.
        """
        overlay = img_bgr.copy()

        for art in articles:
            a_id = art["id"]
            is_selected = (selected_id is None) or (selected_id == a_id)

            hx, hy, hw, hh = art["headline_box"]
            has_text = bool(art.get("headline", "").strip() or art.get("body_text", "").strip())

            if is_selected and selected_id is not None:
                # Active selection: Bright Emerald Green for headline
                color_hl = (30, 220, 60)
                cv2.rectangle(overlay, (hx, hy), (hx + hw, hy + hh), color_hl, 4)
                badge_text = f"#{a_id} HEADLINE" if has_text else f"#{a_id} (NO OCR)"
                cv2.putText(overlay, badge_text, (hx, max(20, hy - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_hl, 2, cv2.LINE_AA)

                # Active body text: Vibrant Cyan
                for bx, by, bw, bh in art["body_boxes"]:
                    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (255, 200, 0), 2)
            else:
                # Non-selected / overview
                color_hl = (220, 100, 40) if has_text else (140, 120, 100)
                cv2.rectangle(overlay, (hx, hy), (hx + hw, hy + hh), color_hl, 2)
                cv2.putText(overlay, f"#{a_id}", (hx + 4, hy + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_hl, 2, cv2.LINE_AA)

                for bx, by, bw, bh in art["body_boxes"]:
                    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (180, 150, 100), 1)

        return overlay

if __name__ == "__main__":
    reader = NewspaperReader()
    sample_news = os.path.join(BASE_DIR, "layout", "News", "0000023720.jpg")
    if os.path.exists(sample_news):
        result = reader.process_newspaper(sample_news)
        print(f"\nProcessed Newspaper: {sample_news}")
        print(f"Total articles detected: {result['total_articles']}")
        for a in result["articles"]:
            hl = a["headline"] if a["headline"] else "[No headline text detected]"
            print(f"\n[Article #{a['id']}] {hl}")
            print(f"  Layout Score: {a['layout_score']*100:.1f}% | Headline Status: {a['headline_status']} | Body Status: {a['body_status']}")
            print(f"  Words: {a['word_count']} | Warnings: {a['warnings']}")
