"""PDF text extraction using Google Cloud Vision OCR."""

import hashlib
import io
from pathlib import Path
from datetime import datetime
from typing import Optional

import fitz  # PyMuPDF for PDF page rendering
from google.cloud import vision

from ..models import Document, DocumentStatus
from ..config import Config
from ..logging_config import logger
from ..exceptions import ExtractionError


class GoogleVisionExtractor:
    """Extracts text from PDF files using Google Cloud Vision OCR.

    This extractor converts PDF pages to images and uses Google Cloud Vision
    for OCR. It's useful for scanned PDFs or documents with complex layouts
    that PyMuPDF can't handle well.

    Requires:
        - google-cloud-vision package
        - GOOGLE_APPLICATION_CREDENTIALS env variable pointing to service account JSON
        - Or default application credentials configured via gcloud CLI
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.extraction_config = self.config.extraction
        self._client = None

    @property
    def client(self) -> vision.ImageAnnotatorClient:
        """Lazy initialization of Vision client."""
        if self._client is None:
            self._client = vision.ImageAnnotatorClient()
        return self._client

    def compute_file_hash(self, pdf_path: Path) -> str:
        """Compute SHA-256 hash of a PDF file for change detection."""
        sha256 = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return f"sha256:{sha256.hexdigest()}"

    def extract_to_markdown(
        self,
        pdf_path: Path,
        document_id: str,
        output_dir: Optional[Path] = None,
        dpi: int = 300,
    ) -> tuple[Document, Path]:
        """
        Extract text from a PDF using Google Cloud Vision OCR.

        Args:
            pdf_path: Path to the PDF file
            document_id: Unique ID for this document
            output_dir: Directory to save markdown files. Defaults to config path.
            dpi: Resolution for page rendering (default: 300)

        Returns:
            Tuple of (Document metadata, Path to markdown file)

        Raises:
            ExtractionError: If extraction fails
        """
        pdf_path = Path(pdf_path)
        output_dir = output_dir or self.config.paths.markdown_dir

        logger.info(
            f"Starting Google Vision OCR extraction: {pdf_path.name}",
            extra={"document_id": document_id, "file_path": str(pdf_path)},
        )

        start_time = datetime.utcnow()

        try:
            # Get file hash for change detection
            file_hash = self.compute_file_hash(pdf_path)

            # Open PDF
            doc = fitz.open(pdf_path)
            page_count = doc.page_count

            # Extract title from filename
            title = self._extract_title_from_filename(pdf_path.name)

            # Process each page with Google Vision OCR
            all_text = []
            for page_num in range(page_count):
                page = doc[page_num]

                # Render page to image
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")

                # Send to Google Vision
                image = vision.Image(content=img_bytes)
                response = self.client.document_text_detection(image=image)

                if response.error.message:
                    raise ExtractionError(
                        f"Google Vision API error: {response.error.message}",
                        document_id=document_id,
                        phase="ocr",
                    )

                page_text = response.full_text_annotation.text if response.full_text_annotation else ""

                # Add page header and text
                all_text.append(f"\n\n<!-- Page {page_num + 1} -->\n\n{page_text}")

                logger.debug(f"Processed page {page_num + 1}/{page_count}")

            doc.close()

            # Combine all pages
            markdown_text = "\n".join(all_text)

            # Create markdown file with frontmatter
            markdown_path = output_dir / f"{document_id}.md"
            markdown_content = self._create_markdown_with_frontmatter(
                document_id=document_id,
                filename=pdf_path.name,
                title=title,
                page_count=page_count,
                file_hash=file_hash,
                content=markdown_text,
            )

            # Write markdown file
            markdown_path.write_text(markdown_content, encoding="utf-8")

            # Create document metadata
            document = Document(
                id=document_id,
                filename=pdf_path.name,
                title=title,
                file_hash=file_hash,
                page_count=page_count,
                status=DocumentStatus.PROCESSING,
                extraction_method="google_vision",
                source_path=str(pdf_path),
                markdown_path=str(markdown_path),
                processed_at=datetime.utcnow(),
            )

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(
                f"Google Vision OCR extraction complete: {pdf_path.name}",
                extra={
                    "document_id": document_id,
                    "duration_ms": duration_ms,
                    "page_count": page_count,
                },
            )

            return document, markdown_path

        except Exception as e:
            logger.error(
                f"Google Vision OCR extraction failed: {pdf_path.name} - {e}",
                extra={"document_id": document_id, "error": str(e)},
            )
            raise ExtractionError(
                f"Failed to extract PDF with Google Vision: {e}",
                document_id=document_id,
                phase="extraction",
                file_path=str(pdf_path),
            ) from e

    def _extract_title_from_filename(self, filename: str) -> str:
        """Extract a clean title from the PDF filename."""
        import re
        name = Path(filename).stem
        name = re.sub(r"^[A-Z]{2,3}\d{2,4}\s*", "", name)
        name = re.sub(r"[_\s]?\d{6}$", "", name)
        name = re.sub(r"[_\s]?\d{4}$", "", name)
        return name.strip()

    def _create_markdown_with_frontmatter(
        self,
        document_id: str,
        filename: str,
        title: str,
        page_count: int,
        file_hash: str,
        content: str,
    ) -> str:
        """Create markdown content with YAML frontmatter."""
        frontmatter = f"""---
document_id: "{document_id}"
filename: "{filename}"
title: "{title}"
page_count: {page_count}
extracted_at: "{datetime.utcnow().isoformat()}Z"
extraction_method: "google_vision"
file_hash: "{file_hash}"
---

"""
        return frontmatter + content
