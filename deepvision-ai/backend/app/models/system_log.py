"""
DeepVision AI — System Log Entity Model
Auditable, structured operational and telemetry event log for system diagnostics.
CRITICAL MANDATE:
Never logs passwords, tokens, API keys, database credentials, or raw uploaded image buffers.
"""

from typing import Optional
from sqlalchemy import String, Float, DateTime, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDPrimaryKeyMixin


class SystemLog(Base, UUIDPrimaryKeyMixin):
    """Platform audit and diagnostic system log entry."""

    __tablename__ = "system_logs"

    level: Mapped[str] = mapped_column(
        String(16),
        index=True,
        nullable=False,
    )
    service: Mapped[str] = mapped_column(
        String(64),
        default="deepvision-core",
        nullable=False,
    )
    endpoint: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )
    request_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        index=True,
        nullable=True,
    )
    job_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        index=True,
        nullable=True,
    )
    processing_time_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )
    error_type: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
    )
    message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    details_json: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True,
        nullable=False,
    )

