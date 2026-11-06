# 🗞️ AI Newspaper Reader – OCR 3 Improved Architecture

An enterprise-grade, research-ready OCR architecture designed to bridge the generalization gap between standard isolated character datasets and complex, multi-column, degraded real-world newspaper scans.

---

## 1. 🎯 Primary Objective

The legacy character-level OCR model achieved high accuracy on synthetic character datasets (~98%) but suffered significant accuracy drop (50–60%) when evaluated on real-world newspaper images due to:
- Complex multi-column reading flows and variable column gutters
- Varied font typography (bold headline banners vs tight body columns)
- Age degradation, paper yellowing, ink bleeds, and scanning perspective skews

This improved architecture (**OCR 3**) introduces a hybrid design combining:
1. **Pretrained Detection & Multi-Column Layout Grouping** (PaddleOCR DBNet + morphology)
2. **Transfer Learning & CTC Sequence Recognition** (`CRNNRecognizer` + `PaddleTextRecognizer`)
3. **Multi-Dataset Management** (Standard OCR, IMPACT PAGE XML, ENP ICDAR, PRImA)
4. **Comprehensive Evaluation Suite** (CER, WER, F1, Exact Match, Normalized Edit Distance, Reading Order Score)

---

## 2. 🏗️ Model Strategy

### A. OCR Detection (`models/paddle_detector.py`)
- Detects arbitrary-oriented text bounding boxes and multi-column bands.
- Distinguishes **HEADLINE**, **COLUMN_BODY**, and **BANNER** regions.
- Handles reading-order sorting (top-to-bottom within column bands).

### B. OCR Recognition (`models/paddle_recognizer.py` & `models/crnn_recognizer.py`)
- Integrates pretrained English recognition weights (`en_PP-OCRv4`, `en_PP-OCRv3`).
- Provides custom CRNN + Bidirectional LSTM + CTC loss architecture with layer freezing for fine-tuning.
- Preserves pretrained base weights before fine-tuning, storing fine-tuned checkpoints in `checkpoints/finetuned_recognition/`.

### C. Alternative Model Comparative Evaluation (`models/alternative_evaluator.py`)
An experimental module to benchmark alternative engines on the exact same test dataset:
- **PaddleOCR** (Default Production Model)
- **EasyOCR** (Baseline Deep Learning Engine)
- **Microsoft TrOCR** (Vision Transformer Encoder-Decoder)
- **DocTR** (Document Text Recognition)

---

## 3. 📂 Dataset Management (`datasets/`)

Supports 4 standard document formats:
1. **Standard OCR Dataset**: Kaggle directory structure (`training_data/<class>/<img_id>.png`).
2. **IMPACT Newspaper Ground Truth**: European libraries newspaper digitization ground truth in PAGE XML.
3. **ENP Dataset**: ICDAR 2015 Electronic Newspaper Processing (ALTO XML format).
4. **PRImA Datasets**: Layout and Text Ground Truth XML.

Features:
- Validates image-text paired samples and detects corrupted images.
- Implements **hash-based leak-free splitting** (`train`, `val`, `test`).
- Curates a separate **Real-World Newspaper Test Split**.

---

## 4. 🧹 Preprocessing & Augmentation (`preprocessing/`)

### Preprocessing (`preprocessing/preprocessor.py`):
- Grayscale conversion
- Noise suppression (Bilateral filter, Median filter)
- Automatic Deskewing via text contour projection & minimum area rectangle
- Contrast enhancement via CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Adaptive Gaussian thresholding
- Resolution normalization (upscaling to standard reading DPI)
- 4-point perspective warp correction

### Realistic Augmentation (`preprocessing/augmentor.py`):
- Gaussian noise & Gaussian blur
- Motion blur (simulating scanner movement)
- Mild random rotation ($\pm 2.5^\circ$)
- Brightness & contrast variations
- JPEG compression artifacts
- Text stroke erosion/dilation (simulating ink bleed and faded print)

---

## 5. 🏋️ Fine-Tuning Pipeline (`training/`)

- PyTorch Connectionist Temporal Classification (`nn.CTCLoss`)
- CNN feature extractor backbone freezing / unfreezing for staged transfer learning
- Cosine Annealing learning rate schedule
- Early stopping with patience tracking
- Checkpoint recovery (resuming interrupted training sessions)
- RAM and VRAM telemetry tracking (`psutil` + `torch.cuda`)
- Separate checkpoint directory: `checkpoints/finetuned_recognition/`

---

## 6. 📊 Evaluation System (`evaluation/`)

Comprehensive metrics implemented in `evaluation/metrics.py`:
- **Character Error Rate (CER)**: Normalized Levenshtein distance on characters.
- **Word Error Rate (WER)**: Normalized edit distance on word tokens.
- **Normalized Edit Distance (NED)**: Character similarity score in $[0.0, 1.0]$.
- **Sequence / Exact Match Accuracy**: Ratio of verbatim ground-truth matches.
- **Token Precision, Recall, and F1 Score**.
- **Layout-Aware Reading Order Score**: Kendall's Tau concordance ranking score.
- **Benchmark Reporter (`evaluation/benchmark_reporter.py`)**: Exports comparison tables in Markdown, CSV, and JSON.

---

## 🚀 Quick Start & CLI Execution

### 1. Run the Full Architecture Demo:
```powershell
python "newspaper reading\ocr 3\run_demo.py"
```

### 2. Run Fine-Tuning Pipeline:
```powershell
python "newspaper reading\ocr 3\training\trainer.py" --epochs 5 --batch_size 16 --lr 0.0005
```

### 3. Run Pipeline on any Newspaper Image or PDF:
```python
from ocr 3.ocr3_pipeline import RobustNewspaperOCRPipeline

pipeline = RobustNewspaperOCRPipeline()
result = pipeline.process_document("path/to/newspaper.pdf")  # or .jpg / .png
print(f"Total articles detected: {result['total_articles']}")
for article in result["articles"]:
    print(f"Headline: {article['headline']}")
    print(f"Body: {article['body_text'][:100]}...\n")
```

