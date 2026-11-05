import os
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union

class NewspaperPreprocessor:
    """
    Robust image preprocessing pipeline for newspaper scans and camera photos:
    - Grayscale conversion
    - Noise filtering
    - Automatic deskewing
    - Contrast enhancement (CLAHE)
    - Adaptive thresholding
    - Resolution normalization
    - Perspective correction
    - Region cropping
    """

    def __init__(self, target_dpi: int = 300, min_line_height: int = 32):
        self.target_dpi = target_dpi
        self.min_line_height = min_line_height

    def to_numpy(self, img: Union[np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(img, Image.Image):
            return np.array(img)
        return img.copy()

    def to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """Converts BGR/RGB to single-channel grayscale."""
        if len(img.shape) == 2:
            return img
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def remove_noise(self, gray: np.ndarray, method: str = "bilateral") -> np.ndarray:
        """Removes paper grain, ink bleed, and scanning artifacts."""
        if method == "bilateral":
            return cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
        elif method == "median":
            return cv2.medianBlur(gray, 3)
        elif method == "gaussian":
            return cv2.GaussianBlur(gray, (3, 3), 0)
        return gray

    def detect_skew_angle(self, gray: np.ndarray) -> float:
        """Estimates skew angle using horizontal text contour projection / minAreaRect."""
        # Thresholding
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Dilate horizontally to connect text lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        # Contours of lines
        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        angles = []
        for c in contours:
            if cv2.contourArea(c) < 300:
                continue
            rect = cv2.minAreaRect(c)
            w, h = rect[1]
            angle = rect[2]
            
            # Normalize angle
            if w < h:
                angle = angle + 90.0
            if angle > 45:
                angle -= 90.0
            if angle < -45:
                angle += 90.0

            if abs(angle) < 25.0:
                angles.append(angle)

        if not angles:
            return 0.0
        return float(np.median(angles))

    def deskew(self, img: np.ndarray, angle: Optional[float] = None) -> Tuple[np.ndarray, float]:
        """Rotates image to align text lines horizontally."""
        gray = self.to_grayscale(img)
        if angle is None:
            angle = self.detect_skew_angle(gray)

        if abs(angle) < 0.2:
            return img, 0.0

        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated, angle

    def enhance_contrast(self, gray: np.ndarray, clip_limit: float = 2.5) -> np.ndarray:
        """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def adaptive_threshold(self, gray: np.ndarray, block_size: int = 19, C: int = 9) -> np.ndarray:
        """Binarizes degraded or unevenly lit newspaper text."""
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C
        )

    def normalize_resolution(self, img: np.ndarray, min_height: int = 1200) -> np.ndarray:
        """Upscales low-resolution scans to standard reading resolution."""
        h, w = img.shape[:2]
        if h < min_height:
            scale = float(min_height) / float(h)
            new_w = int(w * scale)
            return cv2.resize(img, (new_w, min_height), interpolation=cv2.INTER_CUBIC)
        return img

    def correct_perspective(self, img: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Performs four-point perspective warp given detected corner points."""
        # Order points: top-left, top-right, bottom-right, bottom-left
        rect = np.zeros((4, 2), dtype="float32")
        s = corners.sum(axis=1)
        rect[0] = corners[np.argmin(s)]
        rect[2] = corners[np.argmax(s)]
        diff = np.diff(corners, axis=1)
        rect[1] = corners[np.argmin(diff)]
        rect[3] = corners[np.argmax(diff)]

        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        return warped

    def crop_region(self, img: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Crops sub-region with safe boundary clamping. bbox: (x, y, w, h)"""
        x, y, w, h = bbox
        ih, iw = img.shape[:2]
        x1 = max(0, min(x, iw - 1))
        y1 = max(0, min(y, ih - 1))
        x2 = max(x1 + 1, min(x + w, iw))
        y2 = max(y1 + 1, min(y + h, ih))
        return img[y1:y2, x1:x2]

    def preprocess(self, img_input: Union[np.ndarray, Image.Image], deskew_enabled: bool = True) -> np.ndarray:
        """Full end-to-end preprocessing pipeline for newspaper reading."""
        img = self.to_numpy(img_input)
        img = self.normalize_resolution(img)
        gray = self.to_grayscale(img)

        if deskew_enabled:
            gray, _ = self.deskew(gray)

        clean = self.remove_noise(gray)
        enhanced = self.enhance_contrast(clean)
        return enhanced
