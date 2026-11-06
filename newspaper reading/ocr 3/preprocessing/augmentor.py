import random
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class AugmentationConfig:
    p_gaussian_noise: float = 0.35
    p_gaussian_blur: float = 0.30
    p_motion_blur: float = 0.25
    p_rotation: float = 0.40
    max_rotation_angle: float = 2.5
    p_brightness_contrast: float = 0.45
    brightness_range: tuple = (-25, 25)
    contrast_range: tuple = (0.8, 1.25)
    p_jpeg_compression: float = 0.30
    min_jpeg_quality: int = 35
    p_perspective_distortion: float = 0.20
    p_stroke_thickness: float = 0.30

class NewspaperAugmentor:
    """
    Realistic visual data augmentations designed specifically for newspaper OCR training
    without destroying legible character topology.
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.cfg = config or AugmentationConfig()

    def add_gaussian_noise(self, img: np.ndarray, mean: float = 0.0, sigma: float = 12.0) -> np.ndarray:
        noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
        noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy

    def add_gaussian_blur(self, img: np.ndarray, ksize: int = 3) -> np.ndarray:
        return cv2.GaussianBlur(img, (ksize, ksize), 0)

    def add_motion_blur(self, img: np.ndarray, size: int = 4) -> np.ndarray:
        kernel = np.zeros((size, size))
        kernel[int((size - 1) / 2), :] = np.ones(size)
        kernel = kernel / size
        return cv2.filter2D(img, -1, kernel)

    def random_rotation(self, img: np.ndarray, max_angle: float = 2.5) -> np.ndarray:
        angle = random.uniform(-max_angle, max_angle)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Background is white in newspaper text (255)
        border_val = 255 if len(img.shape) == 2 else (255, 255, 255)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=border_val)
        return rotated

    def adjust_brightness_contrast(self, img: np.ndarray, b_range=(-25, 25), c_range=(0.8, 1.25)) -> np.ndarray:
        alpha = random.uniform(*c_range)
        beta = random.uniform(*b_range)
        res = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return res

    def simulate_jpeg_compression(self, img: np.ndarray, quality: int = 40) -> np.ndarray:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        if result:
            return cv2.imdecode(encimg, 1 if len(img.shape) == 3 else 0)
        return img

    def perspective_distortion(self, img: np.ndarray, max_delta: int = 4) -> np.ndarray:
        h, w = img.shape[:2]
        src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dx1 = random.randint(0, max_delta)
        dy1 = random.randint(0, max_delta)
        dx2 = random.randint(-max_delta, 0)
        dy2 = random.randint(0, max_delta)
        dx3 = random.randint(0, max_delta)
        dy3 = random.randint(-max_delta, 0)
        dx4 = random.randint(-max_delta, 0)
        dy4 = random.randint(-max_delta, 0)

        dst = np.float32([
            [0 + dx1, 0 + dy1],
            [w + dx2, 0 + dy2],
            [0 + dx3, h + dy3],
            [w + dx4, h + dy4]
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        border_val = 255 if len(img.shape) == 2 else (255, 255, 255)
        warped = cv2.warpPerspective(img, M, (w, h), borderValue=border_val)
        return warped

    def adjust_text_thickness(self, img: np.ndarray) -> np.ndarray:
        """Simulates heavy ink bleed (dilation) or faded print (erosion)."""
        mode = random.choice(["thicken", "thin"])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        
        # Dark text on light background
        # Erode on image = shrink white background = thickens black text
        # Dilate on image = expands white background = thins black text
        if mode == "thicken":
            return cv2.erode(img, kernel, iterations=1)
        else:
            return cv2.dilate(img, kernel, iterations=1)

    def augment(self, img: np.ndarray) -> np.ndarray:
        """Applies stochastic pipeline of configured augmentations."""
        augmented = img.copy()

        if random.random() < self.cfg.p_rotation:
            augmented = self.random_rotation(augmented, self.cfg.max_rotation_angle)

        if random.random() < self.cfg.p_brightness_contrast:
            augmented = self.adjust_brightness_contrast(
                augmented, self.cfg.brightness_range, self.cfg.contrast_range
            )

        if random.random() < self.cfg.p_stroke_thickness:
            augmented = self.adjust_text_thickness(augmented)

        if random.random() < self.cfg.p_gaussian_blur:
            augmented = self.add_gaussian_blur(augmented, ksize=3)

        if random.random() < self.cfg.p_motion_blur:
            augmented = self.add_motion_blur(augmented, size=3)

        if random.random() < self.cfg.p_gaussian_noise:
            augmented = self.add_gaussian_noise(augmented, sigma=10.0)

        if random.random() < self.cfg.p_jpeg_compression:
            q = random.randint(self.cfg.min_jpeg_quality, 75)
            augmented = self.simulate_jpeg_compression(augmented, quality=q)

        if random.random() < self.cfg.p_perspective_distortion:
            augmented = self.perspective_distortion(augmented)

        return augmented

