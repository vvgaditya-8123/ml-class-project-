import os
import sys
import json
import time
import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Ensure stdout uses UTF-8
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NEWS_DIR = os.path.join(BASE_DIR, "News")
MODEL_OUT = os.path.join(BASE_DIR, "layout_model.joblib")
META_OUT = os.path.join(BASE_DIR, "layout_metadata.json")

# Classes
CLASS_HEADLINE = 0
CLASS_BODY = 1
CLASS_BANNER = 2
CLASS_NAMES = {
    CLASS_HEADLINE: "Headline",
    CLASS_BODY: "Body Text",
    CLASS_BANNER: "Banner / Masthead"
}

def extract_block_features(img_gray, x, y, w, h, page_w, page_h):
    """
    Extracts geometric, stroke, and spatial layout features from a candidate text region.
    """
    rel_h = h / float(page_h)
    rel_w = w / float(page_w)
    aspect = w / float(max(1, h))
    rel_y = y / float(page_h)
    rel_x = x / float(page_w)
    rel_area = (w * h) / float(page_w * page_h)

    # Crop block
    crop = img_gray[y:y+h, x:x+w]
    if crop.size == 0:
        return None

    # Ink / stroke density
    thresh = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ink_density = np.sum(thresh == 255) / float(w * h)

    # Estimate average line height using horizontal projection profile
    proj = np.sum(thresh == 255, axis=1)
    text_rows = np.where(proj > (w * 0.05))[0]
    line_count = 1
    if len(text_rows) > 0:
        # Count transitions to estimate line count
        diffs = np.diff(text_rows)
        gaps = np.sum(diffs > 3)
        line_count = max(1, gaps + 1)
    avg_line_height = float(h) / float(line_count)

    # Column span indicator (newspaper column is ~20% of page width)
    col_span = w / float(page_w * 0.22)

    return [
        rel_h,
        rel_w,
        aspect,
        rel_y,
        rel_x,
        rel_area,
        ink_density,
        avg_line_height,
        line_count,
        col_span
    ]

def collect_training_samples(news_dir, max_pages=60):
    """
    Segments candidate regions from newspaper scans and builds a feature dataset.
    """
    files = [f for f in os.listdir(news_dir) if f.lower().endswith(('.jpg', '.png'))][:max_pages]
    X, y = [], []

    print(f"Collecting layout samples across {len(files)} newspaper pages...")

    for fname in files:
        fpath = os.path.join(news_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue

        page_h, page_w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Adaptive thresholding to capture text strokes
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8)

        # Connect text horizontally into candidate lines/blocks
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 3))
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, h_kernel)

        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw < 50 or bh < 12:
                continue

            feats = extract_block_features(gray, bx, by, bw, bh, page_w, page_h)
            if feats is None:
                continue

            # Ground truth layout labeling rules based on newspaper typographical hierarchy
            # 1. Page Banner / Masthead: spans > 70% width or very large area near the top
            if by < (page_h * 0.15) and (bw > (page_w * 0.65) or bh > 120):
                label = CLASS_BANNER
            # 2. Headline: large font line height or tall box spanning across columns
            elif bh >= 35 or feats[7] >= 28 or (feats[2] > 3.5 and bh >= 28):
                label = CLASS_HEADLINE
            # 3. Body Text: standardized column width and line height < 30px
            elif bh < 32 and feats[7] < 24:
                label = CLASS_BODY
            else:
                # Borderline cases: check column span
                if feats[9] > 1.8 and bh >= 25:
                    label = CLASS_HEADLINE
                else:
                    label = CLASS_BODY

            X.append(feats)
            y.append(label)

    return np.array(X), np.array(y)

def train():
    print("=" * 60)
    print("  NEWS LAYOUT ANALYSIS MODEL TRAINING")
    print("=" * 60)

    t0 = time.time()
    X, y = collect_training_samples(NEWS_DIR, max_pages=80)
    print(f"Extracted {len(X):,} labeled layout regions from newspaper scans.")

    counts = np.bincount(y)
    for c_id, count in enumerate(counts):
        print(f"  Class [{c_id}] {CLASS_NAMES.get(c_id, 'Unknown')}: {count:,} samples")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining Random Forest Layout Classifier...")
    clf.fit(X_train, y_train)

    val_preds = clf.predict(X_val)
    acc = accuracy_score(y_val, val_preds)

    print("\n" + "-" * 40)
    print("  LAYOUT MODEL VALIDATION RESULTS")
    print("-" * 40)
    print(f"  Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_val, val_preds, target_names=[CLASS_NAMES[i] for i in sorted(np.unique(y))]))
    print("-" * 40)

    # Fit final model on 100% of data
    clf.fit(X, y)
    joblib.dump(clf, MODEL_OUT)
    print(f"Saved layout model to {MODEL_OUT}")

    metadata = {
        "classes": CLASS_NAMES,
        "feature_names": [
            "rel_h", "rel_w", "aspect", "rel_y", "rel_x",
            "rel_area", "ink_density", "avg_line_height", "line_count", "col_span"
        ],
        "accuracy": round(float(acc), 4),
        "total_samples": len(X),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved layout metadata to {META_OUT}")

if __name__ == "__main__":
    train()

