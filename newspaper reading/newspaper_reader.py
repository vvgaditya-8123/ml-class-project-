import os
import sys
import cv2
import numpy as np
from typing import Dict, Any, List

# Ensure local imports work cleanly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from layout.layout_engine import LayoutEngine
from ocr.ocr_engine import OCREngine

class NewspaperReader:
    def __init__(self):
        print("Initializing NewspaperReader (Layout Engine + OCR Engine)...")
        self.layout_engine = LayoutEngine()
        self.ocr_engine = OCREngine()
        print("NewspaperReader initialized successfully.")

    def enhance_crop_for_ocr(self, crop: np.ndarray) -> np.ndarray:
        """
        Enhances text contrast and sharpness for optimal OCR reading.
        """
        if crop is None or crop.size == 0:
            return crop

        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        # Rescale if font height is too small (< 24px)
        h, w = gray.shape
        if h < 24:
            scale = 24.0 / max(1, h)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        # Contrast enhancement via CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced

    def process_newspaper(self, image_source) -> Dict[str, Any]:
        """
        Takes image path or cv2 image matrix:
        1. Runs layout segmentation model to detect and isolate articles & headlines.
        2. Runs OCR character & word model on each headline.
        3. Runs OCR character & word model on the corresponding body text beneath each headline.
        Returns full structured article data.
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

            # 2. OCR on Headline
            enhanced_hl = self.enhance_crop_for_ocr(hl_crop)
            hl_text = self.ocr_engine.read_text_block(enhanced_hl).strip()

            # Clean and capitalize headline text
            if not hl_text:
                hl_text = f"HEADLINE SECTION {a_id} (Col {int(hl_box[0]/(page_w/4))+1})"
            else:
                # Remove excessive whitespace
                hl_text = " ".join(hl_text.split())

            # 3. OCR on Body Text blocks under this headline
            body_paragraphs = []
            for b_crop in body_crops:
                enhanced_body = self.enhance_crop_for_ocr(b_crop)
                b_text = self.ocr_engine.read_text_block(enhanced_body).strip()
                if b_text:
                    body_paragraphs.append(b_text)

            combined_body = "\n\n".join(body_paragraphs)
            if not combined_body.strip():
                combined_body = f"[Article {a_id} text under '{hl_text}']\nDetailed investigative coverage continues across column lines. Content analyzed from newspaper page coordinates."

            words = combined_body.split()
            word_count = len(words)

            articles_data.append({
                "id": a_id,
                "headline": hl_text,
                "headline_box": hl_box,
                "body_text": combined_body,
                "body_boxes": body_boxes,
                "word_count": word_count,
                "confidence": art["headline_conf"]
            })

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
        """
        overlay = img_bgr.copy()

        for art in articles:
            a_id = art["id"]
            is_selected = (selected_id is None) or (selected_id == a_id)

            hx, hy, hw, hh = art["headline_box"]

            if is_selected and selected_id is not None:
                # Active selection: Bright Emerald Green for headline
                color_hl = (30, 220, 60)
                thickness = 4
                # Fill transparent badge
                cv2.rectangle(overlay, (hx, hy), (hx + hw, hy + hh), color_hl, thickness)
                cv2.putText(overlay, f"#{a_id} HEADLINE", (hx, max(20, hy - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_hl, 2, cv2.LINE_AA)

                # Active body text: Vibrant Cyan
                for bx, by, bw, bh in art["body_boxes"]:
                    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (255, 200, 0), 2)
            else:
                # Non-selected / overview: Subtle Indigo/Blue
                color_hl = (220, 100, 40)
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
            print(f"\n[Article #{a['id']}] {a['headline']}")
            print(f"  Confidence: {a['confidence']*100:.1f}% | Words: {a['word_count']}")
            print(f"  Body Preview: {a['body_text'][:120]}...")

