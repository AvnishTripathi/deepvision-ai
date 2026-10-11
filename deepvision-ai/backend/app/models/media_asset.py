"""
DeepVision AI — Media Asset Entity Model
Tracks metadata and storage references for uploaded image/video media files.
CRITICAL MANDATE:
Raw media binary files and face images are NEVER stored directly in the database.
They are stored in object/file storage and referenced via storage_key.
"""

from typing import Optional
from sqlalchemy import String, Integer, Float, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDPrimaryKeyMixin


class MediaAsset(Base, UUIDPrimaryKeyMixin):
    """Media asset catalog entry pointing to file/object storage."""

    __tablename__ = "media_assets"

    storage_key: Mapped[str] = mapped_column(
        String(512),
        unique=True,
        index=True,
        nullable=False,
    )
    storage_backend: Mapped[str] = mapped_column(
        String(64),
        default="local",
        nullable=False,
    )
    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    file_size_bytes: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    mime_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )
    media_type: Mapped[str] = mapped_column(
        String(32),  # "image" or "video"
        nullable=False,
    )
    width: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    height: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    duration_seconds: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )
    fps: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

