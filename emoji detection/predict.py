import os
import sys

# Ensure local .venv packages can be resolved even if run with default global python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
venv_site = os.path.join(BASE_DIR, ".venv", "Lib", "site-packages")
if os.path.exists(venv_site) and venv_site not in sys.path:
    sys.path.insert(0, venv_site)

import json
import time
import joblib
import numpy as np
from typing import Dict, Any, List

from preprocess import clean_text, EMOJI_METADATA

# Ensure UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "emoji_model.joblib")
DEFAULT_META_PATH = os.path.join(BASE_DIR, "model_metadata.json")

class EmojiPredictor:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, meta_path: str = DEFAULT_META_PATH):
        self.model_path = model_path
        self.meta_path = meta_path
        self.pipeline = None
        self.metadata = None
        self.classes_map = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Please run train.py first."
            )
        self.pipeline = joblib.load(self.model_path)

        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
                # Parse classes map (keys in json are string integers)
                self.classes_map = {
                    int(k): v for k, v in self.metadata.get("classes", {}).items()
                }
        else:
            # Fallback to EMOJI_METADATA
            self.classes_map = {
                k: {"emoji": v[0], "name": v[1]} for k, v in EMOJI_METADATA.items()
            }

    def predict(self, raw_text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Takes raw text input and returns real-time emoji predictions, confidence scores,
        and execution latency in milliseconds.
        """
        t0 = time.perf_counter()

        cleaned = clean_text(raw_text)
        if not cleaned:
            return {
                "input_text": raw_text,
                "cleaned_text": "",
                "top_emoji": "",
                "top_name": "Start typing to see predictions...",
                "top_confidence": 0.0,
                "suggestions": [],
                "latency_ms": (time.perf_counter() - t0) * 1000
            }

        # Predict probabilities
        probabilities = self.pipeline.predict_proba([cleaned])[0]
        classes = self.pipeline.classes_

        # Sort indices in descending probability order
        ranked_indices = np.argsort(probabilities)[::-1][:top_k]

        suggestions: List[Dict[str, Any]] = []
        for idx in ranked_indices:
            label = int(classes[idx])
            prob = float(probabilities[idx])
            info = self.classes_map.get(label, {"emoji": "❓", "name": f"Emoji {label}"})
            suggestions.append({
                "label": label,
                "emoji": info["emoji"],
                "name": info["name"],
                "score": prob,
                "percentage": f"{prob * 100:.1f}%"
            })

        top = suggestions[0] if suggestions else None
        latency = (time.perf_counter() - t0) * 1000

        return {
            "input_text": raw_text,
            "cleaned_text": cleaned,
            "top_emoji": top["emoji"] if top else "",
            "top_name": top["name"] if top else "",
            "top_confidence": top["score"] if top else 0.0,
            "suggestions": suggestions,
            "latency_ms": latency
        }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "Merry Christmas and happy holidays to everyone!"

    print(f"Loading predictor...")
    predictor = EmojiPredictor()
    result = predictor.predict(query, top_k=5)

    print(f"\nQuery: '{result['input_text']}'")
    print(f"Top Prediction: {result['top_emoji']} ({result['top_name']}) - {result['top_confidence']*100:.1f}%")
    print(f"Latency: {result['latency_ms']:.2f} ms")
    print("\nTop 5 Suggestions:")
    for i, s in enumerate(result['suggestions'], 1):
        bar = "█" * int(s['score'] * 30)
        print(f"  {i}. {s['emoji']} {s['name']:<30} {s['percentage']:>6}  {bar}")

