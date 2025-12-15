"""Parse markdown files with YAML frontmatter."""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import re

import yaml

from ..logging_config import logger
from ..exceptions import TransformationError


@dataclass
class ParsedMarkdown:
    """Parsed markdown document with metadata and content."""

    document_id: str
    filename: str
    title: str
    page_count: int
    extracted_at: str
    extraction_method: str
    file_hash: str
    content: str  # Markdown content without frontmatter


class MarkdownParser:
    """Parses markdown files with YAML frontmatter."""

    FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

    def parse(self, markdown_path: Path) -> ParsedMarkdown:
        """
        Parse a markdown file with YAML frontmatter.

        Args:
            markdown_path: Path to the markdown file

        Returns:
            ParsedMarkdown with metadata and content

        Raises:
            TransformationError: If parsing fails
        """
        markdown_path = Path(markdown_path)

        if not markdown_path.exists():
            raise TransformationError(
                f"Markdown file not found: {markdown_path}",
                phase="parsing",
                file_path=str(markdown_path),
            )

        try:
            text = markdown_path.read_text(encoding="utf-8")

            # Extract frontmatter
            match = self.FRONTMATTER_PATTERN.match(text)
            if not match:
                raise TransformationError(
                    f"No YAML frontmatter found in: {markdown_path}",
                    phase="parsing",
                    file_path=str(markdown_path),
                )

            frontmatter_text = match.group(1)
            content = text[match.end() :].strip()

            # Parse YAML
            metadata = yaml.safe_load(frontmatter_text)

            parsed = ParsedMarkdown(
                document_id=metadata.get("document_id", ""),
                filename=metadata.get("filename", ""),
                title=metadata.get("title", ""),
                page_count=metadata.get("page_count", 0),
                extracted_at=metadata.get("extracted_at", ""),
                extraction_method=metadata.get("extraction_method", ""),
                file_hash=metadata.get("file_hash", ""),
                content=content,
            )

            logger.info(
                f"Parsed markdown: {parsed.title}",
                extra={"document_id": parsed.document_id},
            )

            return parsed

        except yaml.YAMLError as e:
            raise TransformationError(
                f"Invalid YAML frontmatter: {e}",
                phase="parsing",
                file_path=str(markdown_path),
            ) from e
        except Exception as e:
            raise TransformationError(
                f"Failed to parse markdown: {e}",
                phase="parsing",
                file_path=str(markdown_path),
            ) from e

    def parse_batch(self, markdown_paths: list[Path]) -> list[ParsedMarkdown]:
        """Parse multiple markdown files."""
        results = []
        for path in markdown_paths:
            results.append(self.parse(path))
        return results
