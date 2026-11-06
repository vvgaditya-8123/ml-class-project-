# Audit Reproducibility Guide (`AUDIT_README.md`)

This guide provides the exact terminal commands, scripts, and environment configurations required to reproduce all measurements, empirical evaluations, and diagnostic artifacts presented in the **Deep Technical Audit of the Newspaper Reading Module**.

---

## 1. Environment & Prerequisites

### 1.1 Verified Platform Configuration
- **Operating System:** Windows 10/11 (64-bit)
- **Python Version:** 3.13.5 (or compatible 3.10–3.13)
- **Core Packages:** `torch>=2.1.0`, `opencv-python>=4.8.0`, `scikit-learn>=1.4.0`, `joblib>=1.3.0`, `pillow>=10.0.0`, `python-docx>=1.0.0`, `matplotlib>=3.7.0`

### 1.2 Setup Environment
```powershell
# Open terminal in the project root directory:
cd "c:\Users\kamth\Downloads\ml-class-project--main"

# Verify Python version and key packages
python -c "import sys, torch, cv2, sklearn, joblib; print('Python:', sys.version); print('PyTorch:', torch.__version__); print('OpenCV:', cv2.__version__); print('Sklearn:', sklearn.__version__)"
```

---

## 2. Reproducing Model Inspections

### 2.1 Inspect Legacy OCR Model Weights (`char_ocr_model.pth`)
To verify that `char_ocr_model.pth` contains exactly 493,927 parameters across 25 tensor weight keys:
```powershell
python -c "
import torch
state = torch.load('ml-class-project--main/newspaper reading/ocr/char_ocr_model.pth', map_location='cpu')
print('Tensor keys:', len(state))
print('Total parameters:', sum(p.numel() for p in state.values()))
"
```

### 2.2 Inspect Layout Classifier & Feature Importances (`layout_model.joblib`)
To verify the Random Forest configuration (120 estimators, depth 12) and confirm that the top 4 geometric features account for 87.23% of the model's weight:
```powershell
python -c "
import joblib, json
model = joblib.load('ml-class-project--main/newspaper reading/layout/layout_model.joblib')
with open('ml-class-project--main/newspaper reading/layout/layout_metadata.json') as f:
    meta = json.load(f)
for name, imp in sorted(zip(meta['feature_names'], model.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f'  {name:18s}: {imp*100:.2f}%')
"
```

---

## 3. Reproducing Training Sample Collection

### 3.1 Verify Layout Sample Count (Resolving 7,687 vs 5,205)
To reproduce the exact sample collection of 7,687 samples across 80 pages from `layout/News`:
```powershell
python -c "
import sys, os
sys.path.insert(0, 'ml-class-project--main/newspaper reading/layout')
from train_layout import collect_training_samples, NEWS_DIR
X, y = collect_training_samples(NEWS_DIR, max_pages=80)
print('Total samples collected:', len(X))
import numpy as np
for c_id, count in enumerate(np.bincount(y)):
    print(f'  Class {c_id}: {count}')
"
```
*Expected Output:* `Total samples collected: 7687 (Class 0: 5647, Class 1: 1956, Class 2: 84)`.

---

## 4. Reproducing Unit Tests

### 4.1 Run Active Test Suite
```powershell
cd "ml-class-project--main/newspaper reading"
python -m unittest test_newspaper_reading.py -v
cd "../.."
```
*Observation:* All 4 tests report `OK`, demonstrating how the tests silently pass despite missing data and placeholder text.

---

## 5. Reproducing Empirical Multi-Page Diagnostics

### 5.1 Run End-to-End Pipeline on 4 Representative Pages
To execute the pipeline across clean, high-res, mid-res, and alternative scans and generate bounding-box overlays and structured JSON analyses in `reports/audit_artifacts/`:
```powershell
python "C:\Users\kamth\.gemini\antigravity\brain\8431a6bb-3bfa-4cf4-bbba-6c14e8530c9e\scratch\run_audit_diagnostics.py"
```

### 5.2 Verify Diagnostic Artifacts
Inspect the generated files in `reports/audit_artifacts/`:
- `0000023720_overlay.jpg`, `0000023720_analysis.json`
- `00625137_overlay.jpg`, `00625137_analysis.json`
- `01138186a_overlay.jpg`, `01138186a_analysis.json`
- `98065867_overlay.jpg`, `98065867_analysis.json`
- `run_summary.json`

---

## 6. Reproducing OCR 3 Prototype Checks

### 6.1 Verify PaddleOCR Dependency Absence
```powershell
python -c "import paddle; import paddleocr"
```
*Expected Output:* `ModuleNotFoundError: No module named 'paddle'`.

### 6.2 Check Hardcoded Benchmark Literals
Inspect lines 77–82 of `ml-class-project--main/newspaper reading/ocr 3/run_demo.py` to confirm that benchmark numbers are hardcoded Python dictionaries.

