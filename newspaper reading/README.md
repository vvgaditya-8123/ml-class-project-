# 📰 Intelligent Newspaper Reader & Layout OCR Pipeline

A dual-model computer vision and deep learning system that analyzes newspaper scans, automatically detects headlines, associates all body text under each respective headline, and presents them in an interactive GUI.

---

## 🌟 Key Capabilities

1. **Dual AI Models Working Together**:
   - **Model 1: Deep Learning OCR Model (`ocr/char_ocr_model.pth`)**:
     - Trained on 20,628 character images across all 36 alphanumeric classes (`0`–`9`, `A`–`Z`).
     - PyTorch `CharCNN` architecture with 98.81% test accuracy.
     - Segments horizontal lines into character and word crops for text extraction.
   - **Model 2: Visual News Layout Model (`layout/layout_model.joblib`)**:
     - Trained on 188 real scanned newspaper pages.
     - Random Forest classifier (99.80% accuracy) analyzing 10 geometric, typographical, and spatial features.
     - Distinguishes **Headlines**, **Body Text Columns**, and **Banners/Mastheads**.

2. **Hierarchical Article Association**:
   - Accurately links body text columns and paragraphs geometrically located underneath each headline within its column bounds.
   - Automatically determines article boundaries before the next headline or section divider.

3. **Interactive Graphical User Interface (`gui.py`)**:
   - **Headline Directory**: Shows all detected headlines in an interactive list on the left.
   - **Interactive Reader**: Clicking any headline instantly displays that headline and the full text under it.
   - **Visual Layout Map**: Highlights the selected headline (in green) and all of its associated body text blocks (in yellow/cyan) directly on the newspaper image.
   - **One-Click Samples**: Pre-loaded dropdown with sample newspaper scans from `layout/News`.
   - **Export & Copy**: 1-click clipboard copy and text file export.

---

## 📂 File Architecture

```
newspaper reading/
├── ocr/
│   ├── data/                   # 36-class alphanumeric dataset (0-9, A-Z)
│   ├── data2/                  # Supplementary character dataset
│   ├── train_ocr.py            # PyTorch CharCNN training script
│   ├── ocr_engine.py           # Character, line & block OCR inference
│   ├── char_ocr_model.pth      # Trained OCR weights (98.81% accuracy)
│   └── char_classes.json       # Character label mappings
├── layout/
│   ├── News/                   # 188 scanned newspaper page images
│   ├── train_layout.py         # Layout classifier training script
│   ├── layout_engine.py        # Visual layout segmentation & article grouping
│   ├── layout_model.joblib     # Trained Layout model (99.80% accuracy)
│   └── layout_metadata.json    # Features & class metadata
├── newspaper_reader.py         # Integration pipeline combining Layout + OCR
├── gui.py                      # Desktop GUI application
├── launch_gui.bat              # 1-click launcher for Windows
├── test_newspaper_reading.py   # Automated unit tests
└── README.md                   # Documentation
```

---

## 🚀 Running the Application

### Option 1: Double-Click the Launcher
In Windows File Explorer, double-click `launch_gui.bat`.

### Option 2: From Terminal
```powershell
cd "c:\dev\ml project\newspaper reading"
python gui.py
```

---

## 🔄 Re-training the Models

If you add more data or want to re-train the models:

1. **Re-train the OCR Model**:
   ```powershell
   python ocr/train_ocr.py
   ```

2. **Re-train the Layout Model**:
   ```powershell
   python layout/train_layout.py
   ```

3. **Run Automated Unit Tests**:
   ```powershell
   python test_newspaper_reading.py
   ```
