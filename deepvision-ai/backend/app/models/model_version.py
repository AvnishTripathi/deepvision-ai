"""
DeepVision AI — Model Version Entity Model
Tracks registered AI computer vision and tracking models, artifact hashes, and deployment statuses.
"""

from typing import Optional
from sqlalchemy import String, Boolean, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDPrimaryKeyMixin


class ModelVersion(Base, UUIDPrimaryKeyMixin):
    """Catalog of registered AI model checkpoints and architectures."""

    __tablename__ = "model_versions"

    model_name: Mapped[str] = mapped_column(
        String(128),
        index=True,
        nullable=False,
    )
    version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    task_type: Mapped[str] = mapped_column(
        String(64),  # e.g., "face_detection", "landmark_localization", "multi_face_tracking"
        nullable=False,
    )
    file_hash: Mapped[Optional[str]] = mapped_column(
        String(64),  # SHA-256 integrity hash
        nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

