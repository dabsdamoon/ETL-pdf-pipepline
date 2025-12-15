"""SQLite storage for document and chunk metadata."""

import sqlite3
import json
import threading
from pathlib import Path
from typing import Optional
from datetime import datetime

from ..models import Document, DocumentStatus, Chunk, ExtractedImage
from ..config import Config
from ..logging_config import logger
from ..exceptions import LoadError


class SQLiteStore:
    """SQLite storage for document metadata.

    Thread-safe implementation that allows connections to be used across threads.
    Uses a lock for write operations to prevent concurrent write issues.
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.db_path = self.config.paths.sqlite_path
        self._connection = None
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection.

        Uses check_same_thread=False to allow the connection to be used
        from multiple threads (required for FastAPI thread pool).
        """
        if self._connection is None:
            self._connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                title TEXT,
                file_hash TEXT NOT NULL,
                page_count INTEGER,
                status TEXT DEFAULT 'pending',
                extraction_method TEXT,
                fallback_reason TEXT,
                source_path TEXT,
                markdown_path TEXT,
                uploaded_at TEXT,
                processed_at TEXT,
                error_message TEXT
            )
        """)

        # Chunks table (metadata only - text/vectors in LanceDB)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER,
                section_h1 TEXT,
                section_h2 TEXT,
                section_h3 TEXT,
                page_numbers TEXT,
                token_count INTEGER,
                created_at TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)

        # Images table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                page_number INTEGER,
                image_index INTEGER,
                file_path TEXT,
                width INTEGER,
                height INTEGER,
                format TEXT,
                caption TEXT,
                position TEXT,
                created_at TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)

        # Indexes
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_images_document ON images(document_id)"
        )

        conn.commit()
        logger.info(f"SQLite database initialized: {self.db_path}")

    # Document operations
    def insert_document(self, document: Document):
        """Insert a new document."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO documents (
                    id, filename, title, file_hash, page_count, status,
                    extraction_method, fallback_reason, source_path, markdown_path,
                    uploaded_at, processed_at, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    document.id,
                    document.filename,
                    document.title,
                    document.file_hash,
                    document.page_count,
                    document.status.value,
                    document.extraction_method,
                    document.fallback_reason,
                    document.source_path,
                    document.markdown_path,
                    document.uploaded_at.isoformat(),
                    document.processed_at.isoformat() if document.processed_at else None,
                    document.error_message,
                ),
            )
            conn.commit()
            logger.info(f"Inserted document: {document.id}", extra={"document_id": document.id})

    def update_document_status(
        self,
        document_id: str,
        status: DocumentStatus,
        error_message: Optional[str] = None,
    ):
        """Update document status."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE documents
                SET status = ?, error_message = ?, processed_at = ?
                WHERE id = ?
            """,
                (status.value, error_message, datetime.utcnow().isoformat(), document_id),
            )
            conn.commit()

    def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM documents WHERE id = ?", (document_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_document(row)

    def get_document_by_path(self, source_path: str) -> Optional[Document]:
        """Get a document by source path."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM documents WHERE source_path = ?", (source_path,))
        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_document(row)

    def get_document_by_hash(self, file_hash: str) -> Optional[Document]:
        """Get a document by file hash."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM documents WHERE file_hash = ?", (file_hash,))
        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_document(row)

    def list_documents(
        self, status: Optional[DocumentStatus] = None
    ) -> list[Document]:
        """List all documents, optionally filtered by status."""
        conn = self._get_connection()
        cursor = conn.cursor()

        if status:
            cursor.execute(
                "SELECT * FROM documents WHERE status = ?", (status.value,)
            )
        else:
            cursor.execute("SELECT * FROM documents")

        return [self._row_to_document(row) for row in cursor.fetchall()]

    def delete_document(self, document_id: str):
        """Delete a document and all associated data."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            cursor.execute("DELETE FROM images WHERE document_id = ?", (document_id,))
            cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))

            conn.commit()
            logger.info(f"Deleted document: {document_id}", extra={"document_id": document_id})

    def _row_to_document(self, row: sqlite3.Row) -> Document:
        """Convert a database row to a Document object."""
        return Document(
            id=row["id"],
            filename=row["filename"],
            title=row["title"],
            file_hash=row["file_hash"],
            page_count=row["page_count"],
            status=DocumentStatus(row["status"]),
            extraction_method=row["extraction_method"],
            fallback_reason=row["fallback_reason"],
            source_path=row["source_path"],
            markdown_path=row["markdown_path"],
            uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
            processed_at=(
                datetime.fromisoformat(row["processed_at"])
                if row["processed_at"]
                else None
            ),
            error_message=row["error_message"],
        )

    # Chunk operations
    def insert_chunks(self, chunks: list[Chunk]):
        """Insert multiple chunks."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            for chunk in chunks:
                cursor.execute(
                    """
                    INSERT INTO chunks (
                        id, document_id, chunk_index, section_h1, section_h2,
                        section_h3, page_numbers, token_count, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.chunk_index,
                        chunk.section_h1,
                        chunk.section_h2,
                        chunk.section_h3,
                        json.dumps(chunk.page_numbers),
                        chunk.token_count,
                        datetime.utcnow().isoformat(),
                    ),
                )

            conn.commit()
            logger.info(
                f"Inserted {len(chunks)} chunks",
                extra={"chunk_count": len(chunks)},
            )

    def delete_chunks(self, document_id: str):
        """Delete all chunks for a document."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            conn.commit()

    def get_chunks_by_document(self, document_id: str) -> list[dict]:
        """Get all chunks for a document (metadata only).

        Note: Full chunk text is stored in LanceDB. This returns metadata from SQLite.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (document_id,),
        )

        return [
            {
                "id": row["id"],
                "document_id": row["document_id"],
                "chunk_index": row["chunk_index"],
                "section_h1": row["section_h1"],
                "section_h2": row["section_h2"],
                "section_h3": row["section_h3"],
                "page_numbers": json.loads(row["page_numbers"]) if row["page_numbers"] else [],
                "token_count": row["token_count"],
            }
            for row in cursor.fetchall()
        ]

    # Image operations
    def insert_images(self, images: list[ExtractedImage]):
        """Insert multiple images."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            for image in images:
                cursor.execute(
                    """
                    INSERT INTO images (
                        id, document_id, page_number, image_index, file_path,
                        width, height, format, caption, position, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        image.id,
                        image.document_id,
                        image.page_number,
                        image.image_index,
                        image.file_path,
                        image.width,
                        image.height,
                        image.format,
                        image.caption,
                        json.dumps(image.position) if image.position else None,
                        image.created_at.isoformat(),
                    ),
                )

            conn.commit()
            logger.info(f"Inserted {len(images)} images")

    def delete_images(self, document_id: str):
        """Delete all images for a document."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM images WHERE document_id = ?", (document_id,))
            conn.commit()

    def get_images_by_document(self, document_id: str) -> list[ExtractedImage]:
        """Get all images for a document."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM images WHERE document_id = ? ORDER BY page_number, image_index",
            (document_id,),
        )

        return [self._row_to_image(row) for row in cursor.fetchall()]

    def get_image(self, image_id: str) -> Optional[ExtractedImage]:
        """Get an image by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM images WHERE id = ?", (image_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_image(row)

    def _row_to_image(self, row: sqlite3.Row) -> ExtractedImage:
        """Convert a database row to an ExtractedImage object."""
        return ExtractedImage(
            id=row["id"],
            document_id=row["document_id"],
            page_number=row["page_number"],
            image_index=row["image_index"],
            file_path=row["file_path"],
            width=row["width"],
            height=row["height"],
            format=row["format"],
            caption=row["caption"],
            position=json.loads(row["position"]) if row["position"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def close(self):
        """Close the database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
