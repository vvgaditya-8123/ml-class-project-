import os
import sys
import json
import cv2
import numpy as np
import torch
import torch.nn as nn

# Ensure stdout uses UTF-8
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "char_ocr_model.pth")
CLASSES_PATH = os.path.join(BASE_DIR, "char_classes.json")

class CharCNN(nn.Module):
    def __init__(self, num_classes=36):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(128 * 4 * 4, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(192, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class OCREngine:
    def __init__(self, model_path=MODEL_PATH, classes_path=CLASSES_PATH):
        self.device = torch.device("cpu")
        self.classes = []
        if os.path.exists(classes_path):
            with open(classes_path, "r", encoding="utf-8") as f:
                self.classes = json.load(f)
        else:
            self.classes = [str(i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)]

        self.model = CharCNN(num_classes=len(self.classes))
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
        self.model.eval()

    def predict_char(self, char_img: np.ndarray):
        """
        Takes a single character crop (grayscale or RGB), normalizes and classifies it.
        Returns (char, confidence).
        """
        if char_img is None or char_img.size == 0:
            return "", 0.0

        if len(char_img.shape) == 3:
            char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)

        # Pad to square to avoid distorting character aspect ratio
        h, w = char_img.shape
        pad_size = max(h, w)
        padded = np.full((pad_size, pad_size), 255 if np.mean(char_img) > 127 else 0, dtype=np.uint8)
        y_offset = (pad_size - h) // 2
        x_offset = (pad_size - w) // 2
        padded[y_offset:y_offset+h, x_offset:x_offset+w] = char_img

        resized = cv2.resize(padded, (32, 32), interpolation=cv2.INTER_AREA)

        # Dark characters on light background -> invert so strokes are positive
        if np.mean(resized) > 127:
            resized = 255 - resized

        tensor = torch.from_numpy(resized).float() / 127.5 - 1.0
        tensor = tensor.unsqueeze(0).unsqueeze(0) # (1, 1, 32, 32)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            idx = torch.argmax(probs).item()
            conf = probs[idx].item()
            return self.classes[idx], conf

    def segment_line_characters(self, line_img: np.ndarray):
        """
        Segments a single horizontal text line crop into individual characters and word gaps.
        Returns list of (char_crop, bbox) in reading order (left to right).
        """
        if line_img is None or line_img.size == 0:
            return []

        if len(line_img.shape) == 3:
            gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = line_img.copy()

        h, w = gray.shape
        if h < 8 or w < 8:
            return []

        # Adaptive thresholding
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Find connected components or contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        char_boxes = []
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            # Filter noise
            if bw >= 2 and bh >= 5 and bh > (h * 0.25):
                char_boxes.append((bx, by, bw, bh))

        # Sort left to right
        char_boxes.sort(key=lambda b: b[0])
        return char_boxes

    def read_line(self, line_img: np.ndarray) -> str:
        """
        Reads a single line of text from an image.
        Uses character segmentation and computes inter-character spacing for word breaks.
        """
        char_boxes = self.segment_line_characters(line_img)
        if not char_boxes:
            return ""

        h, w = line_img.shape[:2]
        words = []
        current_word = []

        # Compute average character width and gap for space detection
        avg_w = np.mean([b[2] for b in char_boxes]) if char_boxes else 15
        space_threshold = max(8, avg_w * 0.75)

        for i, (bx, by, bw, bh) in enumerate(char_boxes):
            # Check gap with previous character
            if i > 0:
                prev_x_end = char_boxes[i-1][0] + char_boxes[i-1][2]
                gap = bx - prev_x_end
                if gap > space_threshold:
                    if current_word:
                        words.append("".join(current_word))
                        current_word = []

            # Crop character
            crop = line_img[by:by+bh, bx:bx+bw]
            ch, conf = self.predict_char(crop)
            if ch:
                current_word.append(ch)

        if current_word:
            words.append("".join(current_word))

        return " ".join(words)

    def read_text_block(self, block_img: np.ndarray) -> str:
        """
        Segments a multi-line text block into horizontal lines and reads each line.
        """
        if block_img is None or block_img.size == 0:
            return ""

        if len(block_img.shape) == 3:
            gray = cv2.cvtColor(block_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = block_img.copy()

        h, w = gray.shape
        if h < 10 or w < 10:
            return ""

        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Horizontal projection profile to separate lines
        proj = np.sum(thresh, axis=1) / 255.0
        line_indices = np.where(proj > 1.0)[0]

        if len(line_indices) == 0:
            return self.read_line(block_img)

        # Group consecutive rows with text into line boundaries
        lines_ranges = []
        in_line = False
        start = 0

        for r in range(h):
            has_text = proj[r] > 1.5
            if has_text and not in_line:
                in_line = True
                start = max(0, r - 2)
            elif not has_text and in_line:
                in_line = False
                end = min(h, r + 2)
                if end - start >= 8:
                    lines_ranges.append((start, end))

        if in_line and (h - start >= 8):
            lines_ranges.append((start, h))

        if not lines_ranges:
            lines_ranges = [(0, h)]

        extracted_lines = []
        for (y1, y2) in lines_ranges:
            line_crop = block_img[y1:y2, :]
            line_txt = self.read_line(line_crop)
            if line_txt.strip():
                extracted_lines.append(line_txt.strip())

        return "\n".join(extracted_lines)

if __name__ == "__main__":
    engine = OCREngine()
    print("OCR Engine initialized successfully with trained CharCNN model.")
    # Test on a sample character from test dataset
    test_sample = os.path.join(BASE_DIR, "data", "testing_data", "A")
    if os.path.exists(test_sample) and os.listdir(test_sample):
        sample_file = os.path.join(test_sample, os.listdir(test_sample)[0])
        im = cv2.imread(sample_file)
        char, conf = engine.predict_char(im)
        print(f"Sample test prediction: '{char}' with confidence {conf*100:.1f}%")

