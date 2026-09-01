"""OpenAI Files API upload service for supported Chainlit attachments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from attachment_manager import AttachmentMetadata, build_attachment_metadata


SUPPORTED_FILE_EXTENSIONS = frozenset(
    {
        ".pdf",
        ".csv",
        ".tsv",
        ".xls",
        ".xlsx",
        ".docx",
        ".pptx",
        ".txt",
        ".md",
        ".markdown",
        ".json",
    }
)
SUPPORTED_FILE_TYPES_MESSAGE = "PDF, CSV, TSV, XLS, XLSX, DOCX, PPTX, TXT, Markdown, and JSON"
MAX_ATTACHMENT_BYTES = 512 * 1024 * 1024
EXPIRATION_SECONDS = 24 * 60 * 60


class AttachmentUploadError(ValueError):
    """Expected user-facing failure while uploading an attachment to OpenAI."""


async def upload_attachment(
    file_path: str | Path,
    original_filename: str,
    mime_type: str | None,
    api_key: str,
) -> AttachmentMetadata:
    """Upload one supported local file and return metadata with its OpenAI ID."""

    source_path = _validate_local_attachment(file_path, original_filename)
    try:
        file_size_bytes = source_path.stat().st_size
    except OSError as exc:
        raise AttachmentUploadError(
            "The attached file could not be accessed. The general chat is still available."
        ) from exc
    client = AsyncOpenAI(api_key=api_key, timeout=60.0)

    try:
        with source_path.open("rb") as attachment_file:
            uploaded_file = await _create_user_data_file(client, attachment_file)
    except Exception as exc:
        raise AttachmentUploadError(
            "I could not process that attachment. The file may be unsupported, "
            "malformed, or too large. The general chat is still available."
        ) from exc
    finally:
        await client.close()

    file_id = getattr(uploaded_file, "id", None)
    if not isinstance(file_id, str) or not file_id:
        raise AttachmentUploadError(
            "I could not process that attachment. The general chat is still available."
        )

    return build_attachment_metadata(
        openai_file_id=file_id,
        original_filename=original_filename,
        mime_type=mime_type,
        file_size_bytes=file_size_bytes,
    )


async def _create_user_data_file(client: AsyncOpenAI, attachment_file: Any) -> Any:
    """Request a short expiry when the installed client supports it."""

    try:
        return await client.files.create(
            file=attachment_file,
            purpose="user_data",
            expires_after={"anchor": "created_at", "seconds": EXPIRATION_SECONDS},
        )
    except TypeError:
        # Older compatible clients may not expose the optional expires_after field.
        attachment_file.seek(0)
        return await client.files.create(file=attachment_file, purpose="user_data")


def _validate_local_attachment(file_path: str | Path, original_filename: str) -> Path:
    if not original_filename or not original_filename.strip():
        raise AttachmentUploadError("The attachment does not have a usable filename.")

    extension = Path(original_filename).suffix.casefold()
    if extension not in SUPPORTED_FILE_EXTENSIONS:
        raise AttachmentUploadError(
            "This file type is not supported in this phase. Supported file types are "
            f"{SUPPORTED_FILE_TYPES_MESSAGE}."
        )

    source_path = Path(file_path)
    if not source_path.is_file():
        raise AttachmentUploadError(
            "The attached file could not be accessed. The general chat is still available."
        )

    try:
        file_size_bytes = source_path.stat().st_size
    except OSError as exc:
        raise AttachmentUploadError(
            "The attached file could not be accessed. The general chat is still available."
        ) from exc
    if file_size_bytes == 0:
        raise AttachmentUploadError(
            "The attached file is empty. The general chat is still available."
        )
    if file_size_bytes > MAX_ATTACHMENT_BYTES:
        raise AttachmentUploadError(
            "The attached file is too large to process. The general chat is still available."
        )
    return source_path
