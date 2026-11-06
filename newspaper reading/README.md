# 📰 Intelligent Newspaper Reader & Layout OCR Pipeline

A modular, truthful computer vision and deep learning system that analyzes newspaper scans, detects layout structure, associates body text under governing headlines, and presents verified transcriptions in an interactive desktop GUI.

---

## 🌟 Key Capabilities

1. **Truthful, Unfalsified Output Contract (`schema.py`)**:
   - Zero tolerance for placeholder text, synthetic headline sections, or fabricated body copy.
   - Clean data contract with explicit `OCRStatus` (`ok`, `empty`, `error`).
   - Strict separation between **Layout Detection Score** (geometric/spatial classification probability) and **OCR Recognition Score** (character/word transcription confidence).
   - Empty or low-confidence regions produce honest `"empty"` statuses and descriptive warnings while keeping visual bounding boxes inspectable on the page map.

2. **Sequence Text Recognition (`ocr/`)**:
   - Decoupled `TextRecognizer` interface (`ocr/base_recognizer.py`) producing standardized `OCRResult`.
   - `SequenceTextRecognizer` (`ocr/sequence_ocr.py`) operating on full horizontal line crops and projection-profiled text blocks, eliminating brittle character segmentation.
   - Legacy `CharCNN` isolated-character classifier preserved under `ocr/legacy_ocr_engine.py` for historical comparison and benchmark reproducibility.

3. **Column-Aware Layout Segmentation & Grouping (`layout/`)**:
   - `LayoutEngine` (`layout/layout_engine.py`) segments newspaper pages into **Headlines**, **Body Columns**, and **Banners/Mastheads**.
   - Normalized coordinates (`[0.0, 1.0]`) ensure robustness across varied scan resolutions and DPI targets.
   - Unique body block ownership prevents duplicate column assignment across distinct articles.
   - Leak-free document-level train/validation splitting (`layout/train_layout.py`) ensures regions from the same newspaper issue never cross evaluation boundaries.

4. **Multi-Format Ingestion & Advanced OCR 3 Suite (`ocr 3/`)**:
   - PDF ingestion via `pypdfium2` (`ocr 3/pdf_handler.py`) without native Poppler dependencies.
   - Preprocessing pipeline (`ocr 3/preprocessing/preprocessor.py`) with automatic deskewing, bilateral filtering, CLAHE contrast enhancement, and coordinate transform tracking.
   - Standardized OCR evaluation metrics (`ocr 3/evaluation/metrics.py`) providing uncapped character error rate (CER), word error rate (WER), token F1, and Kendall's Tau reading order score.
   - Multi-dataset managers (`ocr 3/datasets/`) supporting Standard OCR, IMPACT PAGE XML, ENP ICDAR ALTO, and PRImA formats.

5. **Responsive Desktop Graphical Interface (`gui.py`)**:
   - **Dark 3-Panel Workspace**: Browse headlines on the left, read verified article text in the center, and inspect visual overlay boxes on the right.
   - **Background Worker Thread**: Non-blocking asynchronous analysis prevents UI freezes during heavy layout and OCR processing.
   - **Honest Score Badges**: Distinct badges for Layout Score and OCR Recognition Score.
   - **Export & Copy**: 1-click clipboard copying and plain-text file export.

---

## 📂 File Architecture

```
newspaper reading/
├── schema.py                   # Data contracts: OCRResult, LayoutRegion, ArticleResult, OCRStatus
├── newspaper_reader.py         # Primary pipeline combining Layout + Sequence OCR
├── gui.py                      # Multi-threaded Tkinter desktop application
├── requirements.txt            # Pinned dependencies (Core + PDF + Optional engines)
├── launch_gui.bat              # 1-click launcher for Windows
├── test_newspaper_reading.py   # Comprehensive automated test suite
├── layout/
│   ├── layout_engine.py        # Visual layout segmentation & article grouping
│   ├── layout_model.joblib     # Trained Random Forest layout model
│   ├── layout_metadata.json    # Feature definitions & class metadata
│   ├── train_layout.py         # Group-split layout classifier training script
│   └── News/                   # Scanned newspaper page images
├── ocr/
│   ├── base_recognizer.py      # Abstract TextRecognizer interface
│   ├── sequence_ocr.py         # Sequence-oriented line and block recognizer
│   ├── ocr_engine.py           # Standardized OCREngine adapter
│   ├── legacy_ocr_engine.py    # Preserved CharCNN engine for comparison
│   ├── char_ocr_model.pth      # Trained legacy character classifier weights
│   ├── char_classes.json       # Character label mappings (0-9, A-Z)
│   └── train_ocr.py            # Legacy character classifier training script
└── ocr 3/                      # Advanced Document AI Research Suite
    ├── ocr3_pipeline.py        # End-to-end PDF/Image pipeline with transform tracking
    ├── pdf_handler.py          # Native PDF rendering via pypdfium2
    ├── run_demo.py             # Diagnostic test runner and metrics verifier
    ├── datasets/               # Loaders: Standard OCR, IMPACT, ENP, PRImA
    ├── evaluation/             # Uncapped CER/WER, Kendall's Tau, BenchmarkReporter
    ├── models/                 # CRNN (CNN+BiLSTM+CTC), Paddle detector/recognizer
    ├── preprocessing/          # Deskew, CLAHE, noise filtering, data augmentor
    └── training/               # Staged transfer learning & CTC fine-tuning
```

---

## 🚀 Installation & Running

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Launch the Desktop GUI
Double-click `launch_gui.bat` in Windows File Explorer, or run:
```powershell
python gui.py
```

### 3. Run Automated Unit Tests
```powershell
python test_newspaper_reading.py -v
```

---

## 📊 Evaluation & Verification

To run verification across the OCR 3 architecture components and verify metric computation without mock benchmarks:
```powershell
python "ocr 3/run_demo.py"
```

To re-train the layout model with leak-free document grouping:
```powershell
python layout/train_layout.py
```
