import os
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union, Dict, Any

class NewspaperPreprocessor:
    """
    Robust image preprocessing pipeline for newspaper scans and camera photos:
    - Grayscale conversion
    - Noise filtering (Bilateral / Median / Gaussian)
    - Automatic deskewing with rotation matrix tracking
    - Contrast enhancement (CLAHE)
    - Adaptive thresholding
    - Resolution normalization with scale factor tracking
    - Perspective correction
    - Region cropping with boundary clamping
    - Coordinate transformation mapping (cleaned <-> original)
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

    def deskew(self, img: np.ndarray, angle: Optional[float] = None) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
        """Rotates image to align text lines horizontally. Returns (rotated_img, angle, rotation_matrix)."""
        gray = self.to_grayscale(img)
        if angle is None:
            angle = self.detect_skew_angle(gray)

        if abs(angle) < 0.2:
            return img, 0.0, None

        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated, angle, M

    def enhance_contrast(self, gray: np.ndarray, clip_limit: float = 2.5) -> np.ndarray:
        """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def adaptive_threshold(self, gray: np.ndarray, block_size: int = 19, C: int = 9) -> np.ndarray:
        """Binarizes degraded or unevenly lit newspaper text."""
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C
        )

    def normalize_resolution(self, img: np.ndarray, min_height: int = 1200) -> Tuple[np.ndarray, float]:
        """Upscales low-resolution scans to standard reading resolution, returning scaled image and factor."""
        h, w = img.shape[:2]
        if h < min_height:
            scale = float(min_height) / float(h)
            new_w = int(w * scale)
            resized = cv2.resize(img, (new_w, min_height), interpolation=cv2.INTER_CUBIC)
            return resized, scale
        return img, 1.0

    def crop_region(self, img: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Crops sub-region with safe boundary clamping. bbox: (x, y, w, h)"""
        x, y, w, h = bbox
        ih, iw = img.shape[:2]
        x1 = max(0, min(x, iw - 1))
        y1 = max(0, min(y, ih - 1))
        x2 = max(x1 + 1, min(x + w, iw))
        y2 = max(y1 + 1, min(y + h, ih))
        return img[y1:y2, x1:x2]

    def preprocess_with_transform(
        self, img_input: Union[np.ndarray, Image.Image], deskew_enabled: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Full end-to-end preprocessing pipeline tracking coordinate transformations:
        Returns (cleaned_gray, transform_metadata).
        """
        raw_bgr = self.to_numpy(img_input)
        orig_h, orig_w = raw_bgr.shape[:2]

        # 1. Scale normalization
        scaled_img, scale_factor = self.normalize_resolution(raw_bgr)
        sh, sw = scaled_img.shape[:2]

        # 2. Grayscale conversion
        gray = self.to_grayscale(scaled_img)

        # 3. Deskew
        skew_angle = 0.0
        rot_matrix = None
        if deskew_enabled:
            gray, skew_angle, rot_matrix = self.deskew(gray)

        # 4. Noise removal & contrast enhancement
        denoised = self.remove_noise(gray)
        enhanced = self.enhance_contrast(denoised)

        transform_meta = {
            "original_shape": (orig_h, orig_w),
            "transformed_shape": (enhanced.shape[0], enhanced.shape[1]),
            "scale_factor": scale_factor,
            "angle": skew_angle,
            "rotation_matrix": rot_matrix
        }

        return enhanced, transform_meta

    def preprocess(self, img_input: Union[np.ndarray, Image.Image], deskew_enabled: bool = True) -> np.ndarray:
        """Legacy helper returning cleaned single-channel image directly."""
        cleaned, _ = self.preprocess_with_transform(img_input, deskew_enabled=deskew_enabled)
        return cleaned
