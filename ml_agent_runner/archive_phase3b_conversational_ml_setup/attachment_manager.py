"""In-memory metadata for OpenAI files attached to each Chainlit chat."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class AttachmentMetadata:
    """Metadata only; no file bytes, extracted text, or API keys are retained."""

    openai_file_id: str
    original_filename: str
    mime_type: str | None
    file_size_bytes: int
    uploaded_at: datetime
    upload_status: str = "uploaded"


class AttachmentManager:
    """Track OpenAI file IDs associated with each in-memory Chainlit chat."""

    def __init__(self) -> None:
        self._attachments: dict[str, list[AttachmentMetadata]] = {}

    def add_many(
        self,
        chainlit_session_id: str,
        attachments: list[AttachmentMetadata],
    ) -> None:
        if attachments:
            self._attachments.setdefault(chainlit_session_id, []).extend(attachments)

    def list_for_session(self, chainlit_session_id: str) -> list[AttachmentMetadata]:
        return list(self._attachments.get(chainlit_session_id, []))

    def remove_session(self, chainlit_session_id: str) -> None:
        self._attachments.pop(chainlit_session_id, None)


attachment_manager = AttachmentManager()


def build_attachment_metadata(
    openai_file_id: str,
    original_filename: str,
    mime_type: str | None,
    file_size_bytes: int,
) -> AttachmentMetadata:
    """Create a timestamped attachment record after a successful upload."""

    return AttachmentMetadata(
        openai_file_id=openai_file_id,
        original_filename=original_filename,
        mime_type=mime_type,
        file_size_bytes=file_size_bytes,
        uploaded_at=datetime.now(UTC),
    )
