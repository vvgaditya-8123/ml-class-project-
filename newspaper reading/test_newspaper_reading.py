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

from ocr.ocr_engine import OCREngine
from layout.layout_engine import LayoutEngine
from newspaper_reader import NewspaperReader

class TestNewspaperReadingPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ocr = OCREngine()
        cls.layout = LayoutEngine()
        cls.reader = NewspaperReader()
        cls.news_dir = os.path.join(BASE_DIR, "layout", "News")

    def test_ocr_character_prediction(self):
        """Verifies OCR model accurately recognizes character crops."""
        test_dir = os.path.join(BASE_DIR, "ocr", "data", "testing_data", "A")
        if os.path.exists(test_dir) and os.listdir(test_dir):
            sample_p = os.path.join(test_dir, os.listdir(test_dir)[0])
            im = cv2.imread(sample_p)
            char, conf = self.ocr.predict_char(im)
            self.assertEqual(char, "A")
            self.assertGreater(conf, 0.70)

    def test_layout_engine_segmentation(self):
        """Verifies layout model segments headline and body text blocks."""
        sample_img = os.path.join(self.news_dir, "0000023720.jpg")
        self.assertTrue(os.path.exists(sample_img))
        img = cv2.imread(sample_img)
        articles = self.layout.segment_layout(img)
        self.assertGreater(len(articles), 0)
        first_art = articles[0]
        self.assertIn("headline_box", first_art)
        self.assertIn("body_boxes", first_art)
        self.assertEqual(len(first_art["headline_box"]), 4)

    def test_end_to_end_reader_pipeline(self):
        """Verifies end-to-end headline detection and text association."""
        sample_img = os.path.join(self.news_dir, "0000023720.jpg")
        result = self.reader.process_newspaper(sample_img)
        self.assertIn("articles", result)
        self.assertGreater(result["total_articles"], 0)

        # Check article structure
        art1 = result["articles"][0]
        self.assertIn("headline", art1)
        self.assertIn("body_text", art1)
        self.assertIn("word_count", art1)
        self.assertGreater(len(art1["headline"]), 0)
        self.assertGreater(len(art1["body_text"]), 0)

    def test_visual_overlay_generation(self):
        """Verifies bounding box overlay rendering on newspaper pages."""
        sample_img = os.path.join(self.news_dir, "0000023720.jpg")
        img = cv2.imread(sample_img)
        result = self.reader.process_newspaper(img)
        overlay = self.reader.render_visual_overlay(img, result["articles"], selected_id=1)
        self.assertEqual(overlay.shape, img.shape)
        # Verify overlay is modified
        self.assertFalse(np.array_equal(overlay, img))

if __name__ == "__main__":
    unittest.main()

