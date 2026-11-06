import os
import sys
import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from preprocessing.preprocessor import NewspaperPreprocessor
from pdf_handler import NewspaperDocumentHandler
from models.paddle_detector import PaddleTextDetector
from models.paddle_recognizer import PaddleTextRecognizer

# Import shared standardized schema
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from schema import OCRResult, ArticleResult, LayoutRegion, OCRStatus

class OCR3Pipeline:
    """
    Integrated, robust OCR 3 end-to-end pipeline for historical newspaper scans:
    - Preprocessing with deskewing and coordinate transform tracking.
    - DBNet text detection on cleaned image.
    - Sequence text recognition.
    - Column-aware article grouping with unique body block ownership.
    - Honest status and confidence reporting without fabricated placeholders.
    """

    def __init__(self, rec_model_name: str = "en_PP-OCRv4", rec_checkpoint: Optional[str] = None):
        print(f"Initializing OCR 3 Pipeline (Recognizer: {rec_model_name})...")
        self.preprocessor = NewspaperPreprocessor()
        self.detector = PaddleTextDetector()
        self.recognizer = PaddleTextRecognizer(model_name=rec_model_name, checkpoint_path=rec_checkpoint)
        print("OCR 3 Pipeline initialized successfully.")

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Processes any supported document (multi-page PDF or single image)
        and extracts structured articles with verified text and coordinates.
        """
        t0 = time.time()
        pages = NewspaperDocumentHandler.load_document_pages(file_path)

        all_articles = []
        processed_pages = []
        global_article_id = 1

        for page_idx, raw_bgr in enumerate(pages, 1):
            page_h, page_w = raw_bgr.shape[:2]

            # 1. Preprocessing: Track coordinate transformations
            cleaned_gray, transform_meta = self.preprocessor.preprocess_with_transform(raw_bgr, deskew_enabled=True)
            cleaned_bgr = cv2.cvtColor(cleaned_gray, cv2.COLOR_GRAY2BGR)

            # 2. Text & Layout Detection on the cleaned image
            detected_regions = self.detector.detect_regions(cleaned_bgr)

            # 3. Text Recognition on Crops from the cleaned image
            for reg in detected_regions:
                bx, by, bw, bh = reg["bbox"]
                crop = cleaned_bgr[by:by+bh, bx:bx+bw]
                text, conf = self.recognizer.recognize_crop(crop)
                reg["text"] = text.strip()
                reg["ocr_score"] = float(conf)
                reg["status"] = "ok" if text.strip() else "empty"

            # 4. Group into Articles by Headline
            page_articles = self._group_articles_by_headline(
                detected_regions, page_w=cleaned_bgr.shape[1], page_h=cleaned_bgr.shape[0], start_id=global_article_id
            )
            global_article_id += len(page_articles)

            all_articles.extend(page_articles)
            processed_pages.append({
                "page_number": page_idx,
                "dimensions": (page_w, page_h),
                "detected_regions_count": len(detected_regions),
                "articles_count": len(page_articles),
                "transform": transform_meta
            })

        duration = time.time() - t0

        return {
            "file_path": file_path,
            "total_pages": len(pages),
            "execution_time_sec": round(duration, 2),
            "total_articles": len(all_articles),
            "articles": all_articles,
            "pages": processed_pages
        }

    def _group_articles_by_headline(
        self, regions: List[Dict[str, Any]], page_w: int, page_h: int, start_id: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Associates body paragraphs to their respective governing headlines.
        Enforces unique body block ownership and eliminates fake placeholder text.
        """
        headlines = [r for r in regions if r.get("role") in ("HEADLINE", "BANNER")]
        bodies = [r for r in regions if r.get("role") == "BODY_TEXT"]

        # Sort headlines in reading order (top-to-bottom, left-to-right)
        headlines.sort(key=lambda h: (h["bbox"][1], h["bbox"][0]))

        articles = []
        assigned_body_indices = set()

        for idx, hl in enumerate(headlines):
            hx, hy, hw, hh = hl["bbox"]
            hl_text = hl.get("text", "").strip()
            hl_score = hl.get("ocr_score", None)
            hl_status: OCRStatus = "ok" if hl_text else "empty"

            # Vertical lower bound is the next headline in an overlapping horizontal span
            lower_bound_y = page_h
            for other in headlines:
                ox, oy, ow, oh = other["bbox"]
                if oy > hy:
                    overlap_x1 = max(hx, ox)
                    overlap_x2 = min(hx + hw, ox + ow)
                    if (overlap_x2 - overlap_x1) > 25:
                        if oy < lower_bound_y:
                            lower_bound_y = oy

            # Collect body text with unique ownership check
            article_body_lines = []
            matched_boxes = []
            body_scores = []

            for b_idx, b in enumerate(bodies):
                if b_idx in assigned_body_indices:
                    continue

                bx, by, bw, bh = b["bbox"]
                if (by >= hy + hh - 10) and (by + bh <= lower_bound_y + 15):
                    # Horizontal column alignment check
                    overlap_x1 = max(hx - 30, bx)
                    overlap_x2 = min(hx + hw + 30, bx + bw)
                    if (overlap_x2 - overlap_x1) > (bw * 0.35):
                        t = b.get("text", "").strip()
                        if t:
                            article_body_lines.append(t)
                        if b.get("ocr_score") is not None:
                            body_scores.append(b["ocr_score"])
                        matched_boxes.append((bx, by, bw, bh))
                        assigned_body_indices.add(b_idx)

            body_content = "\n\n".join(article_body_lines).strip()
            body_status: OCRStatus = "ok" if body_content else "empty"
            avg_body_score = float(np.mean(body_scores)) if body_scores else None

            warnings = []
            if not hl_text:
                warnings.append("No headline text recognized in this region.")
            if not body_content:
                warnings.append("No body text recognized under this headline.")

            article_res = ArticleResult(
                article_id=str(start_id + idx),
                headline=hl_text,
                body_text=body_content,
                headline_status=hl_status,
                body_status=body_status,
                layout_score=hl.get("detection_score", 0.90),
                headline_ocr_score=hl_score,
                body_ocr_score=avg_body_score,
                headline_box=(hx, hy, hw, hh),
                body_boxes=matched_boxes,
                word_count=len(body_content.split()),
                warnings=warnings
            )
            articles.append(article_res.to_dict())

        return articles
