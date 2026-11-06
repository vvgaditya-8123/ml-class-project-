import os
import sys
import unittest
import cv2
import numpy as np

# Ensure UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

OCR3_DIR = os.path.join(BASE_DIR, "ocr 3")
if OCR3_DIR not in sys.path:
    sys.path.insert(0, OCR3_DIR)

from schema import OCRResult, ArticleResult, LayoutRegion, OCRStatus
from ocr.ocr_engine import OCREngine
from layout.layout_engine import LayoutEngine
from newspaper_reader import NewspaperReader
from evaluation.metrics import (
    character_error_rate,
    word_error_rate,
    compute_ocr_metrics_suite,
    layout_reading_order_score
)
from models.crnn_recognizer import VOCAB_LIST, CHAR_TO_IDX

class TestNewspaperReadingPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ocr = OCREngine()
        cls.layout = LayoutEngine()
        cls.reader = NewspaperReader()
        cls.news_dir = os.path.join(BASE_DIR, "layout", "News")

    def test_schema_contracts(self):
        """Verifies standardized schema data contracts for OCR and Layout outputs."""
        ocr_res = OCRResult(text="SAMPLE HEADLINE", status="ok", model_name="SequenceOCR", score=0.92)
        self.assertEqual(ocr_res.text, "SAMPLE HEADLINE")
        self.assertEqual(ocr_res.status, "ok")
        self.assertAlmostEqual(ocr_res.score, 0.92)

        art_res = ArticleResult(
            article_id="1",
            headline="Sample Headline",
            body_text="",
            headline_status="ok",
            body_status="empty",
            layout_score=0.95,
            headline_ocr_score=0.91,
            body_ocr_score=None,
            headline_box=(10, 20, 300, 40),
            body_boxes=[],
            word_count=0,
            warnings=["No body text recognized"]
        )
        data = art_res.to_dict()
        self.assertEqual(data["headline_status"], "ok")
        self.assertEqual(data["body_status"], "empty")
        self.assertIsNone(data["body_ocr_score"])
        self.assertEqual(data["word_count"], 0)
        self.assertIn("No body text recognized", data["warnings"])

    def test_no_fake_fallback_on_blank_crop(self):
        """Verifies that empty or blank images produce status='empty' and NEVER fake strings."""
        blank_white = np.full((60, 400), 255, dtype=np.uint8)
        res = self.ocr.recognize(blank_white)
        self.assertEqual(res.status, "empty")
        self.assertEqual(res.text, "")
        self.assertNotIn("HEADLINE SECTION", res.text)
        self.assertNotIn("Detailed investigative coverage", res.text)

    def test_cer_wer_uncapped_metrics(self):
        """Verifies standard edit rate metrics without false 1.0 capping and correct zero handling."""
        # Exact match
        self.assertEqual(character_error_rate("NEWSPAPER", "NEWSPAPER"), 0.0)
        self.assertEqual(word_error_rate("DAILY NEWS", "DAILY NEWS"), 0.0)

        # Substitution
        self.assertAlmostEqual(character_error_rate("ABCD", "ABCE"), 0.25)

        # Insertions exceeding reference length (CER > 1.0)
        cer_inserted = character_error_rate("A", "AAAAA")
        self.assertGreater(cer_inserted, 1.0)
        self.assertEqual(cer_inserted, 4.0)

        # Empty reference
        self.assertEqual(character_error_rate("", ""), 0.0)
        self.assertEqual(character_error_rate("", "ABC"), 3.0)

        # Micro and Macro metrics suite
        suite = compute_ocr_metrics_suite(["HELLO WORLD"], ["HELLO WORLD"])
        self.assertEqual(suite["mean_cer"], 0.0)
        self.assertEqual(suite["corpus_cer"], 0.0)
        self.assertEqual(suite["exact_match_accuracy"], 1.0)

    def test_reading_order_kendalls_tau(self):
        """Verifies topological reading order score via Kendall's tau."""
        self.assertEqual(layout_reading_order_score([1, 2, 3, 4], [1, 2, 3, 4]), 1.0)
        self.assertEqual(layout_reading_order_score([1, 2, 3, 4], [4, 3, 2, 1]), 0.0)

    def test_unique_body_block_ownership(self):
        """Verifies that body text blocks are uniquely assigned and never duplicated across articles."""
        sample_img = os.path.join(self.news_dir, "0000023720.jpg")
        self.assertTrue(os.path.exists(sample_img), f"Sample scan missing: {sample_img}")
        img = cv2.imread(sample_img)
        articles = self.layout.segment_layout(img)

        assigned_boxes = set()
        for art in articles:
            for b in art["body_boxes"]:
                box_tuple = tuple(b)
                self.assertNotIn(box_tuple, assigned_boxes, "Body block assigned to multiple articles!")
                assigned_boxes.add(box_tuple)

    def test_end_to_end_reader_truthful_reporting(self):
        """Verifies end-to-end newspaper reading reports honest scores without fabricated text."""
        sample_img = os.path.join(self.news_dir, "0000023720.jpg")
        self.assertTrue(os.path.exists(sample_img))
        result = self.reader.process_newspaper(sample_img)

        self.assertIn("articles", result)
        self.assertGreater(result["total_articles"], 0)

        first_art = result["articles"][0]
        self.assertIn("layout_score", first_art)
        self.assertIn("headline_ocr_score", first_art)
        self.assertIn("body_ocr_score", first_art)
        self.assertIn("headline_status", first_art)
        self.assertIn("body_status", first_art)
        self.assertIn("warnings", first_art)

        # Bounding box clamping check
        page_w, page_h = result["dimensions"]
        hx, hy, hw, hh = first_art["headline_box"]
        self.assertGreaterEqual(hx, 0)
        self.assertGreaterEqual(hy, 0)
        self.assertLessEqual(hx + hw, page_w)
        self.assertLessEqual(hy + hh, page_h)

        # Verify NO fake template strings are generated
        for art in result["articles"]:
            self.assertNotIn("HEADLINE SECTION", art["headline"])
            self.assertNotIn("Detailed investigative coverage", art["body_text"])

    def test_visual_overlay_generation(self):
        """Verifies bounding box overlay rendering produces valid modified image with correct shape."""
        sample_img = os.path.join(self.news_dir, "0000023720.jpg")
        img = cv2.imread(sample_img)
        result = self.reader.process_newspaper(img)
        overlay = self.reader.render_visual_overlay(img, result["articles"], selected_id=1)
        self.assertEqual(overlay.shape, img.shape)
        self.assertFalse(np.array_equal(overlay, img))

    def test_legacy_character_classification(self):
        """Tests legacy character recognition when test directory is present, otherwise skips with explicit reason."""
        test_dir = os.path.join(BASE_DIR, "ocr", "data", "testing_data", "A")
        if not os.path.exists(test_dir) or not os.listdir(test_dir):
            self.skipTest("Synthetic character test data is excluded from git tracking via .gitignore; skipped.")
        sample_p = os.path.join(test_dir, os.listdir(test_dir)[0])
        im = cv2.imread(sample_p)
        char, conf = self.ocr.predict_char(im)
        self.assertEqual(char, "A")
        self.assertGreater(conf, 0.70)

if __name__ == "__main__":
    unittest.main()
