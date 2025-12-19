"""Extract module - PDF text and image extraction."""

from .validator import PDFValidator
from .pymupdf_extractor import PyMuPDFExtractor
from .image_extractor import ImageExtractor

# Google Vision OCR extractor (optional - requires google-cloud-vision)
try:
    from .google_vision_extractor import GoogleVisionExtractor
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GoogleVisionExtractor = None
    GOOGLE_VISION_AVAILABLE = False

__all__ = [
    "PDFValidator",
    "PyMuPDFExtractor",
    "ImageExtractor",
    "GoogleVisionExtractor",
    "GOOGLE_VISION_AVAILABLE",
]
