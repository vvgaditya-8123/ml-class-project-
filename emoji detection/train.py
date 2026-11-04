import os
import sys

# Ensure local .venv packages can be resolved even if run with default global python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
venv_site = os.path.join(BASE_DIR, ".venv", "Lib", "site-packages")
if os.path.exists(venv_site) and venv_site not in sys.path:
    sys.path.insert(0, venv_site)

import json
import time
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from preprocess import clean_text, EMOJI_METADATA

# Reconfigure stdout for utf-8
sys.stdout.reconfigure(encoding='utf-8')

TRAIN_CSV = os.path.join(BASE_DIR, "Train.csv")
TEST_CSV = os.path.join(BASE_DIR, "Test.csv")
MAPPING_CSV = os.path.join(BASE_DIR, "Mapping.csv")
MODEL_OUT = os.path.join(BASE_DIR, "emoji_model.joblib")
META_OUT = os.path.join(BASE_DIR, "model_metadata.json")
PRED_OUT = os.path.join(BASE_DIR, "test_predictions.csv")

def build_pipeline():
    """
    High-accuracy multi-scale text feature extraction pipeline:
    - Word n-grams (1 to 3) capturing full phrases & collocations
    - Sublinear TF scaling to damp outlier frequencies
    - Character boundary n-grams (2 to 6) capturing subwords, typos, and keystroke prefixes
    - Calibrated L-BFGS Logistic Regression for optimal multi-class probability estimation
    """
    vectorizer = FeatureUnion([
        ('word', TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=2,
            max_features=55000,
            sublinear_tf=True
        )),
        ('char_wb', TfidfVectorizer(
            ngram_range=(2, 6),
            analyzer='char_wb',
            min_df=3,
            max_features=45000,
            sublinear_tf=True
        ))
    ])

    classifier = LogisticRegression(
        C=1.2,
        max_iter=300,
        random_state=42,
        solver='lbfgs'
    )

    pipeline = Pipeline([
        ('features', vectorizer),
        ('classifier', classifier)
    ])
    return pipeline

def main():
    print("=" * 60)
    print("  EMOJI PREDICTION MODEL TRAINING PIPELINE (HIGH ACCURACY)")
    print("=" * 60)

    # 1. Load Data
    print(f"\n[1/5] Loading datasets from {BASE_DIR}...")
    train_df = pd.read_csv(TRAIN_CSV)
    mapping_df = pd.read_csv(MAPPING_CSV)
    print(f"Loaded {len(train_df):,} training examples across {len(mapping_df)} emoji classes.")

    mapping = {}
    for _, row in mapping_df.iterrows():
        num = int(row['number'])
        emoji_char = row['emoticons']
        name = EMOJI_METADATA.get(num, (emoji_char, f"Emoji {num}"))[1]
        mapping[num] = {
            "emoji": emoji_char,
            "name": name
        }

    # 2. Preprocess Text
    print("\n[2/5] Cleaning and normalizing text with hashtag segmentation & emoticon mapping...")
    t0 = time.time()
    clean_texts = train_df['TEXT'].apply(clean_text)
    labels = train_df['Label'].values
    print(f"Preprocessed {len(clean_texts):,} texts in {time.time()-t0:.2f}s.")

    # 3. Train/Validation Split & Evaluation
    print("\n[3/5] Evaluating model on Stratified Validation Set (85/15 split)...")
    X_train, X_val, y_train, y_val = train_test_split(
        clean_texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    print(f"Train set: {len(X_train):,} samples | Validation set: {len(X_val):,} samples")

    eval_pipeline = build_pipeline()
    t_train = time.time()
    eval_pipeline.fit(X_train, y_train)
    print(f"Validation model trained in {time.time()-t_train:.2f}s.")

    val_probs = eval_pipeline.predict_proba(X_val)
    val_preds = np.argmax(val_probs, axis=1)

    top1_acc = accuracy_score(y_val, val_preds)
    top3_acc = np.mean([y in np.argsort(val_probs[i])[-3:] for i, y in enumerate(y_val)])
    top5_acc = np.mean([y in np.argsort(val_probs[i])[-5:] for i, y in enumerate(y_val)])
    macro_f1 = f1_score(y_val, val_preds, average='macro')
    weighted_f1 = f1_score(y_val, val_preds, average='weighted')

    print("\n" + "-" * 45)
    print("  VALIDATION PERFORMANCE METRICS")
    print("-" * 45)
    print(f"  Top-1 Accuracy : {top1_acc * 100:.2f}%")
    print(f"  Top-3 Accuracy : {top3_acc * 100:.2f}%")
    print(f"  Top-5 Accuracy : {top5_acc * 100:.2f}%")
    print(f"  Macro F1 Score : {macro_f1:.4f}")
    print(f"  Weighted F1    : {weighted_f1:.4f}")
    print("-" * 45)

    # 4. Train Final Production Model on Full Training Set
    print("\n[4/5] Training final production model on 100% of data (70,000 samples)...")
    final_pipeline = build_pipeline()
    t_full = time.time()
    final_pipeline.fit(clean_texts, labels)
    print(f"Final model fitted in {time.time()-t_full:.2f}s.")

    # Save model artifact
    print(f"Saving model to {MODEL_OUT}...")
    joblib.dump(final_pipeline, MODEL_OUT, compress=3)
    model_size_mb = os.path.getsize(MODEL_OUT) / (1024 * 1024)
    print(f"Model saved successfully ({model_size_mb:.2f} MB).")

    # Save metadata
    metadata = {
        "num_classes": len(mapping),
        "classes": mapping,
        "metrics": {
            "top1_accuracy": round(float(top1_acc), 4),
            "top3_accuracy": round(float(top3_acc), 4),
            "top5_accuracy": round(float(top5_acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4)
        },
        "trained_samples": len(train_df),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Metadata saved to {META_OUT}.")

    # 5. Predict on Test.csv if present
    if os.path.exists(TEST_CSV):
        print(f"\n[5/5] Generating predictions on {TEST_CSV}...")
        test_df = pd.read_csv(TEST_CSV)
        clean_test = test_df['TEXT'].apply(clean_text)
        probs = final_pipeline.predict_proba(clean_test)
        preds = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)

        test_df['predicted_label'] = preds
        test_df['predicted_emoji'] = [mapping[p]['emoji'] for p in preds]
        test_df['confidence'] = np.round(confidences, 4)

        test_df.to_csv(PRED_OUT, index=False)
        print(f"Generated predictions for {len(test_df):,} test rows -> {PRED_OUT}")

    print("\n Training Pipeline Completed Successfully! ")

if __name__ == "__main__":
    main()
