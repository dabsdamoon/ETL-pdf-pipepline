"""Image extraction from PDF files."""

from pathlib import Path
from datetime import datetime
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image
import io

from ..models import ExtractedImage
from ..config import Config
from ..logging_config import logger
from ..exceptions import ExtractionError


class ImageExtractor:
    """Extracts images from PDF files."""

    def __init__(self, config: Config = None):
        self.config = config or Config()

    def extract_images(
        self,
        pdf_path: Path,
        document_id: str,
        output_dir: Optional[Path] = None,
        min_width: int = 100,
        min_height: int = 100,
    ) -> list[ExtractedImage]:
        """
        Extract all images from a PDF file.

        Args:
            pdf_path: Path to the PDF file
            document_id: Unique ID for this document
            output_dir: Directory to save images. Defaults to config path.
            min_width: Minimum image width to extract
            min_height: Minimum image height to extract

        Returns:
            List of ExtractedImage metadata objects

        Raises:
            ExtractionError: If extraction fails
        """
        pdf_path = Path(pdf_path)
        output_dir = output_dir or self.config.paths.images_dir

        # Create document-specific directory
        doc_images_dir = output_dir / document_id
        doc_images_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Extracting images from: {pdf_path.name}",
            extra={"document_id": document_id},
        )

        extracted_images = []

        try:
            doc = fitz.open(pdf_path)

            for page_num in range(doc.page_count):
                page = doc[page_num]
                image_list = page.get_images(full=True)

                for img_index, img_info in enumerate(image_list):
                    try:
                        image_meta = self._extract_single_image(
                            doc=doc,
                            img_info=img_info,
                            page_num=page_num,
                            img_index=img_index,
                            document_id=document_id,
                            output_dir=doc_images_dir,
                            min_width=min_width,
                            min_height=min_height,
                        )
                        if image_meta:
                            extracted_images.append(image_meta)

                    except Exception as e:
                        logger.warning(
                            f"Failed to extract image {img_index} from page {page_num}: {e}",
                            extra={"document_id": document_id},
                        )
                        continue

            doc.close()

            logger.info(
                f"Extracted {len(extracted_images)} images from {pdf_path.name}",
                extra={"document_id": document_id},
            )

            return extracted_images

        except Exception as e:
            logger.error(
                f"Image extraction failed: {pdf_path.name} - {e}",
                extra={"document_id": document_id, "error": str(e)},
            )
            raise ExtractionError(
                f"Failed to extract images: {e}",
                document_id=document_id,
                phase="image_extraction",
                file_path=str(pdf_path),
            ) from e

    def _extract_single_image(
        self,
        doc: fitz.Document,
        img_info: tuple,
        page_num: int,
        img_index: int,
        document_id: str,
        output_dir: Path,
        min_width: int,
        min_height: int,
    ) -> Optional[ExtractedImage]:
        """Extract a single image from the PDF."""
        xref = img_info[0]

        # Get image data
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]

        # Load with PIL to get dimensions and validate
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size

        # Skip small images (likely icons, bullets, etc.)
        if width < min_width or height < min_height:
            return None

        # Generate filename
        filename = f"page_{page_num + 1:03d}_img_{img_index + 1:03d}.{image_ext}"
        output_path = output_dir / filename

        # Save image
        img.save(output_path)

        # Create metadata
        return ExtractedImage(
            document_id=document_id,
            page_number=page_num + 1,  # 1-indexed for human readability
            image_index=img_index + 1,
            file_path=str(output_path),
            width=width,
            height=height,
            format=image_ext,
            created_at=datetime.utcnow(),
        )

    def cleanup_images(self, document_id: str, output_dir: Optional[Path] = None):
        """
        Remove all extracted images for a document.

        Args:
            document_id: ID of the document
            output_dir: Base images directory
        """
        output_dir = output_dir or self.config.paths.images_dir
        doc_images_dir = output_dir / document_id

        if doc_images_dir.exists():
            import shutil
            shutil.rmtree(doc_images_dir)
            logger.info(
                f"Cleaned up images for document",
                extra={"document_id": document_id},
            )
