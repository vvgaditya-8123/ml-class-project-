import os
import sys
import unittest
import numpy as np

# Ensure UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

from preprocess import clean_text, EMOJI_METADATA
from predict import EmojiPredictor

class TestEmojiPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "emoji_model.joblib")
        if not os.path.exists(model_path):
            raise unittest.SkipTest("Model not trained yet.")
        cls.predictor = EmojiPredictor()

    def test_text_cleaning(self):
        sample = "Check out @john at https://example.com #Christmas #Party &amp; fun!"
        cleaned = clean_text(sample)
        self.assertNotIn("@john", cleaned)
        self.assertNotIn("https://", cleaned)
        self.assertIn("christmas", cleaned)
        self.assertIn("&", cleaned)

    def test_predictor_response_structure(self):
        res = self.predictor.predict("Merry Christmas to all!", top_k=5)
        self.assertIn("top_emoji", res)
        self.assertIn("top_name", res)
        self.assertIn("top_confidence", res)
        self.assertIn("suggestions", res)
        self.assertEqual(len(res["suggestions"]), 5)
        self.assertLess(res["latency_ms"], 50.0)  # Latency should be sub 50ms (usually < 2ms)

    def test_christmas_prediction(self):
        res = self.predictor.predict("Merry Christmas happy holidays tree santa gift", top_k=5)
        top_emojis = [s["emoji"] for s in res["suggestions"][:3]]
        # 🎄 should be in top 3
        self.assertIn("🎄", top_emojis)

    def test_fire_prediction(self):
        res = self.predictor.predict("This song beat is straight firelit hottest track", top_k=5)
        top_emojis = [s["emoji"] for s in res["suggestions"][:3]]
        # 🔥 should be in top 3
        self.assertIn("🔥", top_emojis)

    def test_love_prediction(self):
        res = self.predictor.predict("I love you so much sweetheart heart", top_k=5)
        top_emojis = [s["emoji"] for s in res["suggestions"][:3]]
        # ❤️ or 💕 or 😍 should be in top 3
        love_emojis = {"❤️", "💕", "😍", "💙", "💜", "😘"}
        self.assertTrue(any(e in love_emojis for e in top_emojis))

if __name__ == "__main__":
    unittest.main()

