import os
import sys
import unittest

# Ensure UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

# Ensure local .venv packages can be resolved
base_dir = os.path.dirname(os.path.abspath(__file__))
venv_site = os.path.join(base_dir, ".venv", "Lib", "site-packages")
if os.path.exists(venv_site) and venv_site not in sys.path:
    sys.path.insert(0, venv_site)

from preprocess import clean_text, EMOJI_METADATA
from predict import EmojiPredictor

class TestEmojiPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_path = os.path.join(base_dir, "emoji_model.joblib")
        if not os.path.exists(model_path):
            raise unittest.SkipTest("Model not trained yet.")
        cls.predictor = EmojiPredictor()

    def test_text_cleaning(self):
        sample = "Check out @john at https://example.com #MerryChristmas #Party &amp; :) fun!"
        cleaned = clean_text(sample)
        self.assertNotIn("@john", cleaned)
        self.assertNotIn("https://", cleaned)
        self.assertIn("christmas", cleaned)
        self.assertIn("emoticon_smile", cleaned)

    def test_predictor_structure_and_speed(self):
        res = self.predictor.predict("Merry Christmas to all!", top_k=5)
        self.assertIn("top_emoji", res)
        self.assertIn("top_name", res)
        self.assertIn("top_confidence", res)
        self.assertIn("suggestions", res)
        self.assertEqual(len(res["suggestions"]), 5)
        self.assertLess(res["latency_ms"], 50.0)

    def test_christmas_prediction(self):
        res = self.predictor.predict("Merry Christmas happy holidays tree santa gift", top_k=5)
        top_emojis = [s["emoji"] for s in res["suggestions"][:3]]
        self.assertIn("🎄", top_emojis)

    def test_fire_prediction(self):
        res = self.predictor.predict("This song beat is straight fire hottest track", top_k=5)
        top_emojis = [s["emoji"] for s in res["suggestions"][:3]]
        self.assertIn("🔥", top_emojis)

    def test_crying_prediction(self):
        res = self.predictor.predict("I am sobbing uncontrollably crying my eyes out broken heart", top_k=5)
        top_emojis = [s["emoji"] for s in res["suggestions"][:3]]
        self.assertIn("😭", top_emojis)

    def test_coffee_prediction(self):
        res = self.predictor.predict("Morning fresh hot brewed espresso latte in the cafe", top_k=5)
        top_emojis = [s["emoji"] for s in res["suggestions"][:3]]
        self.assertIn("☕", top_emojis)

    def test_rocket_prediction(self):
        res = self.predictor.predict("Startup product launch day going straight to the moon liftoff", top_k=5)
        top_emojis = [s["emoji"] for s in res["suggestions"][:3]]
        self.assertIn("🚀", top_emojis)

    def test_gym_prediction(self):
        res = self.predictor.predict("Morning gym workout session heavy deadlifts lifting gains", top_k=5)
        top_emojis = [s["emoji"] for s in res["suggestions"][:3]]
        self.assertIn("💪", top_emojis)

if __name__ == "__main__":
    unittest.main()
