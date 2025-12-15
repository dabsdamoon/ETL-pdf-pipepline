"""PDF text extraction using PyMuPDF (pymupdf4llm)."""

import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional
import re

import fitz  # PyMuPDF
import pymupdf4llm

from ..models import Document, DocumentStatus, ExtractionQualityMetrics
from ..config import Config, ExtractionConfig
from ..logging_config import logger
from ..exceptions import ExtractionError


class PyMuPDFExtractor:
    """Extracts text from PDF files using PyMuPDF and pymupdf4llm."""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.extraction_config = self.config.extraction

    def compute_file_hash(self, pdf_path: Path) -> str:
        """Compute SHA-256 hash of a PDF file for change detection."""
        sha256 = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return f"sha256:{sha256.hexdigest()}"

    def extract_title_from_filename(self, filename: str) -> str:
        """
        Extract a clean title from the PDF filename.

        Examples:
            "EP001 Nutrition During Pregnancy.pdf" -> "Nutrition During Pregnancy"
            "FF633 COVID-19 and Pregnancy.pdf" -> "COVID-19 and Pregnancy"
        """
        # Remove .pdf extension
        name = Path(filename).stem

        # Remove common prefixes like "EP001", "FF633", etc.
        name = re.sub(r"^[A-Z]{2,3}\d{2,4}\s*", "", name)

        # Remove date suffixes like "_042022", "122020", etc.
        name = re.sub(r"[_\s]?\d{6}$", "", name)
        name = re.sub(r"[_\s]?\d{4}$", "", name)

        return name.strip()

    def extract_to_markdown(
        self,
        pdf_path: Path,
        document_id: str,
        output_dir: Optional[Path] = None,
    ) -> tuple[Document, Path]:
        """
        Extract text from a PDF and save as a markdown file.

        Args:
            pdf_path: Path to the PDF file
            document_id: Unique ID for this document
            output_dir: Directory to save markdown files. Defaults to config path.

        Returns:
            Tuple of (Document metadata, Path to markdown file)

        Raises:
            ExtractionError: If extraction fails
        """
        pdf_path = Path(pdf_path)
        output_dir = output_dir or self.config.paths.markdown_dir

        logger.info(
            f"Starting extraction: {pdf_path.name}",
            extra={"document_id": document_id, "file_path": str(pdf_path)},
        )

        start_time = datetime.utcnow()

        try:
            # Get file hash for change detection
            file_hash = self.compute_file_hash(pdf_path)

            # Open PDF to get metadata
            doc = fitz.open(pdf_path)
            page_count = doc.page_count
            doc.close()

            # Extract text as markdown using pymupdf4llm
            markdown_text = pymupdf4llm.to_markdown(str(pdf_path))

            # Extract title
            title = self.extract_title_from_filename(pdf_path.name)

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
                extraction_method="pymupdf",
                source_path=str(pdf_path),
                markdown_path=str(markdown_path),
                processed_at=datetime.utcnow(),
            )

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(
                f"Extraction complete: {pdf_path.name}",
                extra={
                    "document_id": document_id,
                    "duration_ms": duration_ms,
                    "page_count": page_count,
                },
            )

            return document, markdown_path

        except Exception as e:
            logger.error(
                f"Extraction failed: {pdf_path.name} - {e}",
                extra={"document_id": document_id, "error": str(e)},
            )
            raise ExtractionError(
                f"Failed to extract PDF: {e}",
                document_id=document_id,
                phase="extraction",
                file_path=str(pdf_path),
            ) from e

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
extraction_method: "pymupdf"
file_hash: "{file_hash}"
---

"""
        return frontmatter + content

    def analyze_quality(self, text: str, page_count: int) -> ExtractionQualityMetrics:
        """
        Analyze the quality of extracted text.

        Args:
            text: Extracted text content
            page_count: Number of pages in the document

        Returns:
            ExtractionQualityMetrics with computed values
        """
        if not text or page_count == 0:
            return ExtractionQualityMetrics(
                chars_per_page=0,
                words_per_page=0,
                avg_word_length=0,
                whitespace_ratio=1.0,
                non_ascii_ratio=0,
                empty_pages_ratio=1.0,
            )

        words = text.split()
        word_count = len(words)
        char_count = len(text)

        return ExtractionQualityMetrics(
            chars_per_page=char_count / page_count,
            words_per_page=word_count / page_count,
            avg_word_length=sum(len(w) for w in words) / max(word_count, 1),
            whitespace_ratio=text.count(" ") / max(char_count, 1),
            non_ascii_ratio=sum(1 for c in text if ord(c) > 127) / max(char_count, 1),
            empty_pages_ratio=0,
        )

    def check_quality(
        self, metrics: ExtractionQualityMetrics
    ) -> tuple[bool, Optional[str]]:
        """
        Check if extraction quality meets thresholds.

        Args:
            metrics: Quality metrics to check

        Returns:
            Tuple of (is_acceptable, reason_if_not)
        """
        config = self.extraction_config

        if metrics.chars_per_page < config.min_chars_per_page:
            return False, f"Low char density: {metrics.chars_per_page:.1f}/page"

        if metrics.words_per_page < config.min_words_per_page:
            return False, f"Low word count: {metrics.words_per_page:.1f}/page"

        if metrics.avg_word_length < config.min_avg_word_length:
            return False, f"Short words: {metrics.avg_word_length:.1f} avg"

        if metrics.avg_word_length > config.max_avg_word_length:
            return False, f"No word boundaries: {metrics.avg_word_length:.1f} avg"

        if metrics.non_ascii_ratio > config.max_non_ascii_ratio:
            return False, f"High non-ASCII: {metrics.non_ascii_ratio:.2%}"

        return True, None
