"""
DeepVision AI — Analysis Job Entity Model
Tracks asynchronous image and video computer vision processing jobs.
"""

from typing import Optional
from datetime import datetime
from sqlalchemy import String, Integer, DateTime, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, UUIDPrimaryKeyMixin, TimestampMixin


class AnalysisJob(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Asynchronous vision pipeline execution job record."""

    __tablename__ = "analysis_jobs"

    user_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    job_type: Mapped[str] = mapped_column(
        String(32),  # "image_detection", "image_analysis", "video_tracking"
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(32),  # "queued", "processing", "completed", "failed", "cancelled"
        default="queued",
        index=True,
        nullable=False,
    )
    input_media_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("media_assets.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    output_media_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("media_assets.id", ondelete="SET NULL"),
        nullable=True,
    )
    progress: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    total_frames: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    processed_frames: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    request_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        index=True,
        nullable=True,
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationships
    user = relationship("User", foreign_keys=[user_id], lazy="selectin")
    input_media = relationship("MediaAsset", foreign_keys=[input_media_id], lazy="selectin")
    output_media = relationship("MediaAsset", foreign_keys=[output_media_id], lazy="selectin")

