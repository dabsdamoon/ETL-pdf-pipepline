"""Custom exceptions for the ETL pipeline."""


class PipelineError(Exception):
    """Base exception for pipeline errors."""

    def __init__(
        self,
        message: str,
        document_id: str = None,
        phase: str = None,
        file_path: str = None,
    ):
        self.document_id = document_id
        self.phase = phase
        self.file_path = file_path
        super().__init__(message)


class ValidationError(PipelineError):
    """Raised when PDF validation fails."""

    pass


class ExtractionError(PipelineError):
    """Raised when PDF extraction fails."""

    pass


class TransformationError(PipelineError):
    """Raised when chunking or embedding fails."""

    pass


class LoadError(PipelineError):
    """Raised when database loading fails."""

    pass
