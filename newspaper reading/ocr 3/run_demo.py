import os
import sys
import time
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from preprocessing.preprocessor import NewspaperPreprocessor
from pdf_handler import NewspaperPDFHandler
from evaluation.metrics import compute_ocr_metrics_suite
from evaluation.benchmark_reporter import BenchmarkReporter
from models.alternative_evaluator import AlternativeModelEvaluator

def main():
    print("=" * 70)
    print("  AI Newspaper Reader - OCR 3 Architecture Verification Runner")
    print("=" * 70)

    # 1. Image Preprocessing Module
    print("\n[1/4] Testing Image Preprocessing Module (Deskew & Normalization)...")
    preprocessor = NewspaperPreprocessor()
    test_img_path = os.path.join(os.path.dirname(BASE_DIR), "layout", "News", "0000023720.jpg")
    
    if os.path.exists(test_img_path):
        raw_bgr = cv2.imread(test_img_path)
        print(f"  ✓ Loaded test page: {os.path.basename(test_img_path)} ({raw_bgr.shape[1]}x{raw_bgr.shape[0]})")
        cleaned, transform = preprocessor.preprocess_with_transform(raw_bgr, deskew_enabled=True)
        print(f"  ✓ Preprocessing complete. Cleaned shape: {cleaned.shape}, Deskew angle: {transform.get('angle', 0.0):.2f}°")
    else:
        print(f"  ⚠ Test page not found at: {test_img_path}")

    # 2. PDF Handler Integration
    print("\n[2/4] Testing PDF Ingestion Module...")
    pdf_handler = NewspaperPDFHandler()
    print(f"  ✓ PDF Handler operational (pypdfium2: {pdf_handler.pypdfium2_available})")

    # 3. Alternative Model Availability
    print("\n[3/4] Checking OCR Engine Adapters...")
    evaluator = AlternativeModelEvaluator()
    for engine_name, info in evaluator.available_engines.items():
        print(f"  • {engine_name}: {info['status']}")

    # 4. Evaluation Suite Verification (Real Metric Calculation)
    print("\n[4/4] Verifying Metric Computation Suite (Uncapped CER / WER)...")
    sample_refs = [
        "THE DAILY CHRONICLE INTERNATIONAL EDITION",
        "GLOBAL MARKETS RALLY FOLLOWING ECONOMIC SUMMIT",
        "PRICES RISE MODERATELY IN FOURTH QUARTER REPORT"
    ]
    sample_hyps = [
        "THE DAILY CHRONICLE INTERNATIONAL EDITION",
        "GLOBAL MARKETS RALLY FOLLOWING ECONOMIC SUMMIT",
        "PRICES ROSE MODERATELY IN FOURTH QUARTER REPORT"
    ]
    metrics = compute_ocr_metrics_suite(sample_refs, sample_hyps)
    print(f"  ✓ Evaluated Metrics on Sample Predictions (3 pairs):")
    print(f"    • Mean CER: {metrics['mean_cer']*100:.2f}% | Corpus CER: {metrics['corpus_cer']*100:.2f}%")
    print(f"    • Mean WER: {metrics['mean_wer']*100:.2f}% | Corpus WER: {metrics['corpus_wer']*100:.2f}%")
    print(f"    • Exact Match Accuracy: {metrics['exact_match_accuracy']*100:.2f}%")
    print(f"    • Mean F1 Score: {metrics['mean_f1']:.4f}")

    print("\n  [Notice] End-of-run benchmark summary: No paired ground-truth test manifest configured.")
    print("  To run full evaluation across engines, supply a verified line-level ground-truth dataset.")

if __name__ == "__main__":
    main()
