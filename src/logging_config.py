"""Structured JSON logging configuration."""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Formatter that outputs JSON-structured log records."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        extra_fields = [
            "document_id",
            "duration_ms",
            "chunk_count",
            "error",
            "phase",
            "file_path",
            "page_count",
        ]
        for key in extra_fields:
            if hasattr(record, key):
                log_data[key] = getattr(record, key)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def setup_logging(
    log_dir: Optional[Path] = None,
    log_level: int = logging.INFO,
    logger_name: str = "etl_pipeline",
) -> logging.Logger:
    """
    Set up structured logging with JSON file output and console output.

    Args:
        log_dir: Directory for log files. Defaults to ./logs
        log_level: Logging level. Defaults to INFO
        logger_name: Name of the logger

    Returns:
        Configured logger instance
    """
    if log_dir is None:
        log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # Clear existing handlers
    logger.handlers.clear()

    # File handler with JSON format
    file_handler = logging.FileHandler(log_dir / "pipeline.json")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(JSONFormatter())

    # Console handler for human-readable output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Default logger instance
logger = setup_logging()
