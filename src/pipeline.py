"""Main ETL pipeline orchestration."""

import uuid
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

from tqdm import tqdm

from .config import Config
from .models import Document, DocumentStatus, ValidationResult
from .logging_config import logger
from .exceptions import PipelineError, ValidationError, ExtractionError

from .extract import PDFValidator, PyMuPDFExtractor, ImageExtractor
from .transform import MarkdownParser, HybridChunker, Embedder
from .load import SQLiteStore, LanceDBStore


class Pipeline:
    """
    Main ETL pipeline for PDF document processing.

    Flow:
    1. Validate PDF
    2. Extract to Markdown file (checkpoint)
    3. Extract images
    4. Parse markdown and chunk text
    5. Generate embeddings
    6. Load to databases (SQLite + LanceDB)
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()

        # Initialize components
        self.validator = PDFValidator(self.config.extraction)
        self.extractor = PyMuPDFExtractor(self.config)
        self.image_extractor = ImageExtractor(self.config)
        self.markdown_parser = MarkdownParser()
        self.chunker = HybridChunker(self.config.chunking)
        self.embedder = Embedder(self.config.embedding)
        self.sqlite_store = SQLiteStore(self.config)
        self.lancedb_store = LanceDBStore(self.config)

    def process_document(self, pdf_path: Path) -> str:
        """
        Process a single PDF document through the full pipeline.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Document ID

        Raises:
            PipelineError: If any step fails (stop-on-failure mode)
        """
        pdf_path = Path(pdf_path)
        document_id = str(uuid.uuid4())

        logger.info(
            f"Starting pipeline for: {pdf_path.name}",
            extra={"document_id": document_id, "file_path": str(pdf_path)},
        )

        start_time = datetime.utcnow()

        try:
            # Step 1: Validate
            validation_result = self.validator.validate(pdf_path)
            if validation_result != ValidationResult.VALID:
                raise ValidationError(
                    f"PDF validation failed: {validation_result.value}",
                    document_id=document_id,
                    phase="validation",
                    file_path=str(pdf_path),
                )

            # Step 2: Extract to markdown
            document, markdown_path = self.extractor.extract_to_markdown(
                pdf_path, document_id
            )

            # Step 3: Extract images
            images = self.image_extractor.extract_images(pdf_path, document_id)

            # Step 4: Parse markdown and chunk
            parsed = self.markdown_parser.parse(markdown_path)
            chunks = self.chunker.chunk(
                parsed.content,
                document_id,
                document.title,
            )

            # Step 5: Generate embeddings
            chunks = self.embedder.embed_chunks(chunks)

            # Step 6: Load to databases
            self.sqlite_store.insert_document(document)
            self.sqlite_store.insert_chunks(chunks)
            self.sqlite_store.insert_images(images)
            self.lancedb_store.insert_chunks(chunks)

            # Update status to completed
            self.sqlite_store.update_document_status(document_id, DocumentStatus.COMPLETED)

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(
                f"Pipeline complete: {pdf_path.name}",
                extra={
                    "document_id": document_id,
                    "duration_ms": duration_ms,
                    "chunk_count": len(chunks),
                },
            )

            return document_id

        except PipelineError:
            # Re-raise pipeline errors (stop-on-failure)
            raise

        except Exception as e:
            # Wrap unexpected errors
            logger.error(
                f"Pipeline failed: {pdf_path.name} - {e}",
                extra={"document_id": document_id, "error": str(e)},
            )
            raise PipelineError(
                f"Unexpected error: {e}",
                document_id=document_id,
                phase="unknown",
                file_path=str(pdf_path),
            ) from e

    def process_batch(
        self,
        pdf_paths: list[Path],
        progress: bool = True,
    ) -> list[str]:
        """
        Process multiple PDF documents.

        Args:
            pdf_paths: List of paths to PDF files
            progress: Show progress bar

        Returns:
            List of document IDs for successfully processed documents

        Raises:
            PipelineError: On first failure (stop-on-failure mode)
        """
        document_ids = []

        iterator = tqdm(pdf_paths, desc="Processing PDFs") if progress else pdf_paths

        for pdf_path in iterator:
            doc_id = self.process_document(pdf_path)
            document_ids.append(doc_id)

        # Create FTS index after batch load
        self.lancedb_store.create_fts_index()

        logger.info(f"Batch complete: {len(document_ids)} documents processed")
        return document_ids

    def process_directory(
        self,
        directory: Optional[Path] = None,
        progress: bool = True,
    ) -> list[str]:
        """
        Process all PDFs in a directory.

        Args:
            directory: Path to directory with PDFs. Defaults to config.paths.pdf_dir
            progress: Show progress bar

        Returns:
            List of document IDs
        """
        directory = directory or self.config.paths.pdf_dir
        pdf_paths = list(directory.glob("*.pdf"))

        logger.info(f"Found {len(pdf_paths)} PDFs in {directory}")

        return self.process_batch(pdf_paths, progress=progress)

    def process_new_documents(self, progress: bool = True) -> list[str]:
        """
        Process only new or changed documents (incremental).

        Returns:
            List of document IDs for newly processed documents
        """
        pdf_dir = self.config.paths.pdf_dir
        pdf_paths = list(pdf_dir.glob("*.pdf"))

        new_paths = []
        for pdf_path in pdf_paths:
            file_hash = self.extractor.compute_file_hash(pdf_path)
            existing = self.sqlite_store.get_document_by_hash(file_hash)

            if existing is None:
                # New document
                new_paths.append(pdf_path)
            elif existing.status == DocumentStatus.FAILED:
                # Retry failed document
                self.delete_document(existing.id)
                new_paths.append(pdf_path)

        if not new_paths:
            logger.info("No new documents to process")
            return []

        logger.info(f"Found {len(new_paths)} new/changed documents")
        return self.process_batch(new_paths, progress=progress)

    def reprocess_from_markdown(self, document_id: str):
        """
        Re-transform a document from its markdown checkpoint.

        Useful for re-chunking or re-embedding without re-extracting.

        Args:
            document_id: ID of the document to reprocess
        """
        document = self.sqlite_store.get_document(document_id)
        if document is None:
            raise PipelineError(
                f"Document not found: {document_id}",
                document_id=document_id,
                phase="reprocess",
            )

        markdown_path = Path(document.markdown_path)
        if not markdown_path.exists():
            raise PipelineError(
                f"Markdown file not found: {markdown_path}",
                document_id=document_id,
                phase="reprocess",
            )

        logger.info(
            f"Reprocessing from markdown: {document.title}",
            extra={"document_id": document_id},
        )

        # Delete old chunks
        self.sqlite_store.delete_chunks(document_id)
        self.lancedb_store.delete_by_document(document_id)

        # Re-transform
        parsed = self.markdown_parser.parse(markdown_path)
        chunks = self.chunker.chunk(parsed.content, document_id, document.title)
        chunks = self.embedder.embed_chunks(chunks)

        # Re-load
        self.sqlite_store.insert_chunks(chunks)
        self.lancedb_store.insert_chunks(chunks)

        logger.info(
            f"Reprocessed document: {document.title}",
            extra={"document_id": document_id, "chunk_count": len(chunks)},
        )

    def delete_document(self, document_id: str):
        """
        Delete a document and all associated data.

        Args:
            document_id: ID of the document to delete
        """
        document = self.sqlite_store.get_document(document_id)

        if document:
            # Delete markdown file
            if document.markdown_path:
                markdown_path = Path(document.markdown_path)
                if markdown_path.exists():
                    markdown_path.unlink()

            # Delete images directory
            images_dir = self.config.paths.images_dir / document_id
            if images_dir.exists():
                shutil.rmtree(images_dir)

        # Delete from databases
        self.lancedb_store.delete_by_document(document_id)
        self.sqlite_store.delete_document(document_id)

        logger.info(f"Deleted document: {document_id}", extra={"document_id": document_id})

    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        documents = self.sqlite_store.list_documents()
        chunk_count = self.lancedb_store.count()

        by_status = {}
        for doc in documents:
            status = doc.status.value
            by_status[status] = by_status.get(status, 0) + 1

        return {
            "total_documents": len(documents),
            "total_chunks": chunk_count,
            "by_status": by_status,
        }

    def close(self):
        """Close all database connections."""
        self.sqlite_store.close()
        self.lancedb_store.close()
