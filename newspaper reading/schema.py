import os
from dataclasses import dataclass, field
from typing import Literal, Tuple, List, Optional, Dict, Any

OCRStatus = Literal["ok", "empty", "error"]

@dataclass
class OCRResult:
    """Standardized result of text recognition on an image region."""
    text: str
    status: OCRStatus
    model_name: str
    score: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LayoutRegion:
    """Standardized representation of a detected layout bounding box or polygon."""
    region_id: str
    role: str  # "headline", "body_column", "caption", "image", "masthead", "banner", "divider"
    bbox_xyxy: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in original page pixel coordinates
    normalized_bbox: Tuple[float, float, float, float]  # (nx1, ny1, nx2, ny2) in [0.0, 1.0]
    detection_score: Optional[float] = None
    page_id: str = "page_1"
    column_index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ArticleResult:
    """Structured article with explicit separation between layout and OCR scores."""
    article_id: str
    headline: str
    body_text: str
    headline_status: OCRStatus
    body_status: OCRStatus
    layout_score: Optional[float]
    headline_ocr_score: Optional[float]
    body_ocr_score: Optional[float]
    headline_box: Tuple[int, int, int, int]  # (x, y, w, h)
    body_boxes: List[Tuple[int, int, int, int]]  # list of (x, y, w, h)
    word_count: int
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes article result for UI consumption and JSON export."""
        return {
            "id": int(self.article_id) if self.article_id.isdigit() else self.article_id,
            "article_id": self.article_id,
            "headline": self.headline,
            "body_text": self.body_text,
            "headline_status": self.headline_status,
            "body_status": self.body_status,
            "layout_score": self.layout_score,
            "headline_ocr_score": self.headline_ocr_score,
            "body_ocr_score": self.body_ocr_score,
            # Provide backward-compatible confidence field explicitly labeled as layout_score
            "confidence": self.layout_score if self.layout_score is not None else 0.0,
            "headline_box": self.headline_box,
            "body_boxes": self.body_boxes,
            "word_count": self.word_count,
            "warnings": self.warnings,
            "metadata": self.metadata
        }
