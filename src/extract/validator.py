"""PDF validation before extraction."""

from pathlib import Path
import fitz  # PyMuPDF

from ..models import ValidationResult
from ..config import ExtractionConfig
from ..logging_config import logger


class PDFValidator:
    """Validates PDF files before processing."""

    def __init__(self, config: ExtractionConfig = None):
        self.config = config or ExtractionConfig()

    def validate(self, pdf_path: Path) -> ValidationResult:
        """
        Validate a PDF file for processing.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            ValidationResult indicating if the file is valid or the reason it's invalid
        """
        pdf_path = Path(pdf_path)

        # Check file exists
        if not pdf_path.exists():
            logger.warning(f"PDF file not found: {pdf_path}")
            return ValidationResult.FILE_NOT_FOUND

        # Check file size
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        if size_mb > self.config.max_file_size_mb:
            logger.warning(
                f"PDF too large: {size_mb:.2f}MB > {self.config.max_file_size_mb}MB",
                extra={"file_path": str(pdf_path)},
            )
            return ValidationResult.FILE_TOO_LARGE

        # Try to open the PDF
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.warning(
                f"Failed to open PDF (corrupted?): {e}",
                extra={"file_path": str(pdf_path)},
            )
            return ValidationResult.CORRUPTED

        try:
            # Check if password protected
            if doc.needs_pass:
                logger.warning(
                    f"PDF is password protected: {pdf_path}",
                    extra={"file_path": str(pdf_path)},
                )
                return ValidationResult.PASSWORD_PROTECTED

            # Check page count
            if doc.page_count == 0:
                logger.warning(
                    f"PDF is empty (0 pages): {pdf_path}",
                    extra={"file_path": str(pdf_path)},
                )
                return ValidationResult.EMPTY

            if doc.page_count > self.config.max_page_count:
                logger.warning(
                    f"PDF has too many pages: {doc.page_count} > {self.config.max_page_count}",
                    extra={"file_path": str(pdf_path), "page_count": doc.page_count},
                )
                return ValidationResult.TOO_MANY_PAGES

            logger.info(
                f"PDF validated successfully: {pdf_path.name}",
                extra={"file_path": str(pdf_path), "page_count": doc.page_count},
            )
            return ValidationResult.VALID

        finally:
            doc.close()

    def validate_batch(self, pdf_paths: list[Path]) -> dict[Path, ValidationResult]:
        """
        Validate multiple PDF files.

        Args:
            pdf_paths: List of paths to PDF files

        Returns:
            Dictionary mapping paths to their validation results
        """
        results = {}
        for path in pdf_paths:
            results[path] = self.validate(path)
        return results

    def get_valid_pdfs(self, pdf_paths: list[Path]) -> list[Path]:
        """
        Filter to only valid PDF files.

        Args:
            pdf_paths: List of paths to PDF files

        Returns:
            List of paths that passed validation
        """
        results = self.validate_batch(pdf_paths)
        return [path for path, result in results.items() if result == ValidationResult.VALID]
