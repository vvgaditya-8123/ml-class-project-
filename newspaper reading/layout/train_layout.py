import os
import sys
import time
import json
import joblib
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Ensure local imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NEWS_DIR = os.path.join(BASE_DIR, "News")
MODEL_OUT = os.path.join(BASE_DIR, "layout_model.joblib")
META_OUT = os.path.join(BASE_DIR, "layout_metadata.json")

# Classes: 0: Body Text, 1: Headline, 2: Page Banner / Masthead
CLASS_BODY = 0
CLASS_HEADLINE = 1
CLASS_BANNER = 2
CLASS_NAMES = {0: "BODY_TEXT", 1: "HEADLINE", 2: "BANNER"}

def extract_block_features(gray: np.ndarray, x: int, y: int, w: int, h: int, page_w: int, page_h: int):
    """
    Extracts 10 layout geometry & texture features:
    [rel_h, rel_w, aspect, rel_y, rel_x, rel_area, ink_density, avg_line_height, line_count, col_span]
    """
    crop = gray[y:y+h, x:x+w]
    if crop.size == 0:
        return None

    rel_h = float(h) / float(page_h)
    rel_w = float(w) / float(page_w)
    aspect = float(w) / float(max(1, h))
    rel_y = float(y) / float(page_h)
    rel_x = float(x) / float(page_w)
    rel_area = float(w * h) / float(page_w * page_h)

    # Ink density (foreground pixel ratio)
    thresh = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ink_density = float(np.count_nonzero(thresh)) / float(max(1, w * h))

    # Horizontal projection to estimate text lines inside block
    h_proj = np.sum(thresh, axis=1) / 255.0
    gaps = 0
    in_gap = False
    for val in h_proj:
        if val < 2.0 and not in_gap:
            in_gap = True
            gaps += 1
        elif val >= 2.0 and in_gap:
            in_gap = False

    line_count = max(1, gaps + 1)
    avg_line_height = float(h) / float(line_count)
    col_span = w / float(max(1, page_w * 0.22))

    return [
        rel_h, rel_w, aspect, rel_y, rel_x,
        rel_area, ink_density, avg_line_height, line_count, col_span
    ]

def extract_page_samples(fpath: str):
    """Extracts features and heuristic labels for a single page."""
    img = cv2.imread(fpath)
    if img is None:
        return [], []

    page_h, page_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8)
    kernel_w = max(15, min(35, int(page_w * 0.02)))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, h_kernel)

    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page_X, page_y = [], []

    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw < 50 or bh < 12:
            continue
        if bw > (page_w * 0.98) and bh > (page_h * 0.85):
            continue

        feats = extract_block_features(gray, bx, by, bw, bh, page_w, page_h)
        if feats is None:
            continue

        # Heuristic typographical rules for bootstrapping baseline model
        if by < (page_h * 0.15) and (bw > (page_w * 0.65) or bh > 120):
            label = CLASS_BANNER
        elif bh >= 35 or feats[7] >= 28 or (feats[2] > 3.5 and bh >= 28):
            label = CLASS_HEADLINE
        elif bh < 32 and feats[7] < 24:
            label = CLASS_BODY
        else:
            label = CLASS_HEADLINE if (feats[9] > 1.8 and bh >= 25) else CLASS_BODY

        page_X.append(feats)
        page_y.append(label)

    return page_X, page_y

def train():
    print("=" * 60)
    print("  NEWS LAYOUT ANALYSIS MODEL TRAINING (LEAK-FREE GROUP SPLIT)")
    print("=" * 60)

    t0 = time.time()
    if not os.path.exists(NEWS_DIR):
        raise FileNotFoundError(f"Newspaper news directory not found at: {NEWS_DIR}")

    # Use deterministic sorted list
    all_files = sorted([f for f in os.listdir(NEWS_DIR) if f.lower().endswith(('.jpg', '.png'))])
    if not all_files:
        raise RuntimeError(f"No newspaper images found in: {NEWS_DIR}")

    # Leak-free splitting: split by document/page ID, NOT by pooled regions
    train_files, val_files = train_test_split(all_files[:80], test_size=0.20, random_state=42)
    print(f"Split {len(all_files[:80])} pages into {len(train_files)} train pages and {len(val_files)} val pages.")

    X_train, y_train = [], []
    for f in train_files:
        px, py = extract_page_samples(os.path.join(NEWS_DIR, f))
        X_train.extend(px)
        y_train.extend(py)

    X_val, y_val = [], []
    for f in val_files:
        px, py = extract_page_samples(os.path.join(NEWS_DIR, f))
        X_val.extend(px)
        y_val.extend(py)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)

    print(f"Collected {len(X_train):,} train regions and {len(X_val):,} validation regions.")

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
    val_acc = accuracy_score(y_val, val_preds)

    print("\n" + "=" * 50)
    print(f"  Validation Accuracy (Independent Pages): {val_acc * 100:.2f}%")
    print("=" * 50)
    print("\nClassification Report (Held-out Pages):")
    target_names = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))]
    print(classification_report(y_val, val_preds, target_names=target_names, digits=4))

    # Save model and metadata
    joblib.dump(clf, MODEL_OUT)
    print(f"Saved trained layout model -> {MODEL_OUT}")

    metadata = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 120,
        "features": [
            "rel_h", "rel_w", "aspect", "rel_y", "rel_x",
            "rel_area", "ink_density", "avg_line_height", "line_count", "col_span"
        ],
        "classes": CLASS_NAMES,
        "val_accuracy": float(val_acc),
        "split_strategy": "group_document_leak_free",
        "train_pages_count": len(train_files),
        "val_pages_count": len(val_files),
        "trained_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved layout metadata -> {META_OUT}")

if __name__ == "__main__":
    train()
