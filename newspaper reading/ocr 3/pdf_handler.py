import os
from typing import List, Union
import cv2
import numpy as np

class NewspaperDocumentHandler:
    """
    Accepts both Newspaper PDFs and standard images (JPEG, PNG, TIFF, BMP, WebP)
    and extracts high-resolution page arrays for downstream layout detection and OCR.
    Uses pypdfium2 for native cross-platform PDF rendering without poppler dependencies.
    """

    SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    SUPPORTED_PDF_EXTS = {".pdf"}

    @property
    def pypdfium2_available(self) -> bool:
        try:
            import pypdfium2
            return True
        except ImportError:
            return False

    @classmethod
    def load_document_pages(cls, file_path: str, render_dpi: int = 200) -> List[np.ndarray]:
        """Loads all document pages as BGR numpy images."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document file not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext in cls.SUPPORTED_PDF_EXTS:
            return cls._render_pdf_pages(file_path, render_dpi)
        elif ext in cls.SUPPORTED_IMAGE_EXTS:
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError(f"Failed to decode image file: {file_path}")
            return [img]
        else:
            raise ValueError(f"Unsupported file format '{ext}'. Supported: PDFs and images ({cls.SUPPORTED_IMAGE_EXTS})")

    @classmethod
    def _render_pdf_pages(cls, pdf_path: str, dpi: int) -> List[np.ndarray]:
        pages = []
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(pdf_path)
            # Default scale is 72 dpi; scale = dpi / 72.0
            scale = dpi / 72.0
            for page in pdf:
                pil_image = page.render(scale=scale).to_pil()
                # Convert PIL RGB to OpenCV BGR
                bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                pages.append(bgr)
            return pages
        except ImportError:
            raise ImportError("PDF rendering requires 'pypdfium2'. Install via 'pip install pypdfium2'.")

# Alias for backwards compatibility
NewspaperPDFHandler = NewspaperDocumentHandler
