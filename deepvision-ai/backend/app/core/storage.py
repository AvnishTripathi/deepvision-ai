"""
DeepVision AI — Media Storage Abstraction (Step 5.19)
Provides decoupled media storage interface preparing DeepVision AI
for migration between local filesystem, AWS S3, Google Cloud Storage, or Azure Blob.

CRITICAL SECURITY RULE:
Never expose raw server filesystem paths to external API clients.
All media references must use opaque storage keys or download route URIs.
"""

import os
import shutil
import logging
from abc import ABC, abstractmethod
from typing import Optional, BinaryIO
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger("deepvision.storage")


class BaseStorageBackend(ABC):
    """Abstract interface for media storage backends."""

    @abstractmethod
    def store_file(self, source_path: str, destination_key: str) -> str:
        """
        Stores a file into the storage system.
        Returns: opaque storage key identifier.
        """
        pass

    @abstractmethod
    def get_local_path(self, storage_key: str) -> Optional[str]:
        """
        Resolves storage key to an absolute local file path for local processing.
        """
        pass

    @abstractmethod
    def exists(self, storage_key: str) -> bool:
        """Checks whether a media object exists in storage."""
        pass

    @abstractmethod
    def delete(self, storage_key: str) -> bool:
        """Deletes a stored object by key."""
        pass


class LocalStorageBackend(BaseStorageBackend):
    """
    Local filesystem storage provider storing artifacts under configured storage directory.
    Prepares clear foundation for future S3 / GCS / Azure Blob drivers.
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = os.path.abspath(base_dir or settings.STORAGE_DIR)
        os.makedirs(self.base_dir, exist_ok=True)
        # Create partitioned subdirectories
        self.videos_dir = os.path.join(self.base_dir, "videos")
        self.outputs_dir = os.path.join(self.base_dir, "outputs")
        os.makedirs(self.videos_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)

    def store_file(self, source_path: str, destination_key: str) -> str:
        """Copies file to storage location under destination_key."""
        dest_path = os.path.join(self.base_dir, destination_key)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(source_path, dest_path)
        logger.debug("Stored file %s as key %s", source_path, destination_key)
        return destination_key

    def get_local_path(self, storage_key: str) -> Optional[str]:
        """Returns physical absolute path for local disk storage."""
        candidate = os.path.join(self.base_dir, storage_key)
        if os.path.exists(candidate):
            return candidate
        return None

    def exists(self, storage_key: str) -> bool:
        candidate = os.path.join(self.base_dir, storage_key)
        return os.path.exists(candidate)

    def delete(self, storage_key: str) -> bool:
        candidate = os.path.join(self.base_dir, storage_key)
        if os.path.exists(candidate):
            try:
                os.remove(candidate)
                return True
            except OSError as err:
                logger.warning("Failed to delete stored object %s: %s", storage_key, err)
                return False
        return False


# Singleton storage provider
_storage_backend: Optional[BaseStorageBackend] = None


def get_storage_backend() -> BaseStorageBackend:
    """Returns application storage backend singleton."""
    global _storage_backend
    if _storage_backend is None:
        _storage_backend = LocalStorageBackend()
    return _storage_backend
