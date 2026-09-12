class IngestionError(Exception):
    """Base error for an expected ingestion failure."""


class UnsupportedFormatError(IngestionError):
    """Raised when no parser is registered for a file type."""


class ParseError(IngestionError):
    """Raised when a supported document cannot be parsed safely."""


class EmptyDocumentError(IngestionError):
    """Raised when no indexable text remains after cleaning."""


class FileTooLargeError(IngestionError):
    """Raised when an input exceeds the configured size limit."""

