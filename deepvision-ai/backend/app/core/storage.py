"""
DeepVision AI — Production Storage Architecture
Provides a decoupled storage abstraction separating:
- Temporary processing files (scratch/ephemeral frames)
- Persistent media assets (consented uploaded files)
- Generated analytical results (annotated outputs, reports)

Supports:
- LocalStorage: Local disk filesystem storage with segregated partitions
- ObjectStorage: Cloud object storage abstraction (Amazon S3, Google Cloud Storage,
  Azure Blob Storage) with zero-credential local emulation fallback.

CRITICAL SECURITY & PRIVACY RULE:
- Never expose raw server filesystem paths to external API clients.
- All media references must use opaque storage keys or download route URIs.
"""

import os
import shutil
import logging
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Optional, BinaryIO

from app.core.config import settings

logger = logging.getLogger("deepvision.storage")


class StorageCategory(str, Enum):
    """Categorized storage namespaces."""
    TEMPORARY = "temp"       # Scratch/ephemeral frame extraction & temporary buffers
    MEDIA = "media"          # User-uploaded persistent media assets
    RESULTS = "results"      # Generated visual outputs & analytical artifacts


class BaseStorageBackend(ABC):
    """Abstract interface for media and artifact storage backends."""

    @abstractmethod
    def store_file(
        self,
        source_path: str,
        destination_key: str,
        category: StorageCategory = StorageCategory.MEDIA,
    ) -> str:
        """Stores a file into the storage system under destination_key."""
        pass

    @abstractmethod
    def store_bytes(
        self,
        data: bytes,
        destination_key: str,
        category: StorageCategory = StorageCategory.MEDIA,
    ) -> str:
        """Stores binary bytes into the storage system."""
        pass

    @abstractmethod
    def get_local_path(
        self,
        storage_key: str,
        category: StorageCategory = StorageCategory.MEDIA,
    ) -> Optional[str]:
        """Resolves storage key to an absolute local file path for local processing."""
        pass

    @abstractmethod
    def exists(
        self,
        storage_key: str,
        category: StorageCategory = StorageCategory.MEDIA,
    ) -> bool:
        """Checks whether a media object exists in storage."""
        pass

    @abstractmethod
    def delete(
        self,
        storage_key: str,
        category: StorageCategory = StorageCategory.MEDIA,
    ) -> bool:
        """Deletes a stored object by key."""
        pass

    @abstractmethod
    def get_temp_dir(self) -> str:
        """Returns dedicated directory for temporary video processing files."""
        pass


class LocalStorageBackend(BaseStorageBackend):
    """
    Local filesystem storage provider with segregated directories for:
    - Temporary processing files
    - Persistent media assets
    - Generated analytical results
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = os.path.abspath(base_dir or settings.STORAGE_DIR)
        os.makedirs(self.base_dir, exist_ok=True)

        # Segregated physical directories
        self.temp_dir = os.path.join(self.base_dir, "temp")
        self.media_dir = os.path.join(self.base_dir, "media")
        self.results_dir = os.path.join(self.base_dir, "results")

        # Backward compatibility directories for existing video tracking jobs
        self.videos_dir = os.path.join(self.base_dir, "videos")
        self.outputs_dir = os.path.join(self.base_dir, "outputs")

        for d in (self.temp_dir, self.media_dir, self.results_dir, self.videos_dir, self.outputs_dir):
            os.makedirs(d, exist_ok=True)

    def _resolve_target_path(self, storage_key: str, category: StorageCategory) -> str:
        """Resolves key to target directory while preserving backward compatibility."""
        clean_key = storage_key.lstrip("/\\")

        # If key already specifies legacy prefix (e.g. videos/... or outputs/...)
        if clean_key.startswith("videos/") or clean_key.startswith("videos\\"):
            return os.path.join(self.base_dir, clean_key)
        if clean_key.startswith("outputs/") or clean_key.startswith("outputs\\"):
            return os.path.join(self.base_dir, clean_key)

        category_map = {
            StorageCategory.TEMPORARY: self.temp_dir,
            StorageCategory.MEDIA: self.media_dir,
            StorageCategory.RESULTS: self.results_dir,
        }
        target_dir = category_map.get(category, self.media_dir)
        return os.path.join(target_dir, clean_key)

    def store_file(
        self,
        source_path: str,
        destination_key: str,
        category: StorageCategory = StorageCategory.MEDIA,
    ) -> str:
        dest_path = self._resolve_target_path(destination_key, category)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(source_path, dest_path)
        logger.debug("Stored file %s as key %s (%s)", source_path, destination_key, category.value)
        return destination_key

    def store_bytes(
        self,
        data: bytes,
        destination_key: str,
        category: StorageCategory = StorageCategory.MEDIA,
    ) -> str:
        dest_path = self._resolve_target_path(destination_key, category)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(data)
        logger.debug("Stored %d bytes as key %s (%s)", len(data), destination_key, category.value)
        return destination_key

    def get_local_path(
        self,
        storage_key: str,
        category: StorageCategory = StorageCategory.MEDIA,
    ) -> Optional[str]:
        # Check primary resolved path
        primary_path = self._resolve_target_path(storage_key, category)
        if os.path.exists(primary_path):
            return primary_path

        # Fallback to direct relative resolution under base_dir for legacy keys
        legacy_path = os.path.join(self.base_dir, storage_key.lstrip("/\\"))
        if os.path.exists(legacy_path):
            return legacy_path

        return None

    def exists(
        self,
        storage_key: str,
        category: StorageCategory = StorageCategory.MEDIA,
    ) -> bool:
        return self.get_local_path(storage_key, category) is not None

    def delete(
        self,
        storage_key: str,
        category: StorageCategory = StorageCategory.MEDIA,
    ) -> bool:
        path = self.get_local_path(storage_key, category)
        if path and os.path.exists(path):
            try:
                os.remove(path)
                logger.debug("Deleted stored object: %s", storage_key)
                return True
            except OSError as err:
                logger.warning("Failed to delete stored object %s: %s", storage_key, err)
                return False
        return False

    def get_temp_dir(self) -> str:
        return self.temp_dir


# Alias for clean object naming
LocalStorage = LocalStorageBackend


class ObjectStorageBackend(BaseStorageBackend):
    """
    Object Storage abstraction preparing DeepVision AI for future integration with:
    - Amazon Web Services (AWS S3)
    - Google Cloud Storage (GCS)
    - Microsoft Azure Blob Storage
    
    Provides transparent local emulation fallback in development when cloud credentials
    are not supplied, ensuring zero cloud dependency or authentication barriers locally.
    """

    def __init__(
        self,
        provider: str = "emulated",
        bucket_name: str = "deepvision-media",
        local_fallback: Optional[LocalStorageBackend] = None,
    ):
        self.provider = provider.lower()  # "s3", "gcs", "azure", "emulated"
        self.bucket_name = bucket_name
        self._local_backend = local_fallback or LocalStorageBackend()
        logger.info(
            "Initialized ObjectStorageBackend (Provider: %s, Bucket: %s, Cloud Credentials Required: %s)",
            self.provider,
            self.bucket_name,
            "No (Local Emulation Active)" if self.provider == "emulated" else "Yes",
        )

    def store_file(
        self,
        source_path: str,
        destination_key: str,
        category: StorageCategory = StorageCategory.MEDIA,
    ) -> str:
        if self.provider == "emulated":
            return self._local_backend.store_file(source_path, destination_key, category)
        # Cloud providers (boto3 / google-cloud-storage / azure-storage-blob) will execute here
        raise NotImplementedError(f"Cloud provider '{self.provider}' driver requires active cloud credentials.")

    def store_bytes(
        self,
        data: bytes,
        destination_key: str,
        category: StorageCategory = StorageCategory.MEDIA,
    ) -> str:
        if self.provider == "emulated":
            return self._local_backend.store_bytes(data, destination_key, category)
        raise NotImplementedError(f"Cloud provider '{self.provider}' driver requires active cloud credentials.")

    def get_local_path(
        self,
        storage_key: str,
        category: StorageCategory = StorageCategory.MEDIA,
    ) -> Optional[str]:
        if self.provider == "emulated":
            return self._local_backend.get_local_path(storage_key, category)
        return None

    def exists(
        self,
        storage_key: str,
        category: StorageCategory = StorageCategory.MEDIA,
    ) -> bool:
        if self.provider == "emulated":
            return self._local_backend.exists(storage_key, category)
        return False

    def delete(
        self,
        storage_key: str,
        category: StorageCategory = StorageCategory.MEDIA,
    ) -> bool:
        if self.provider == "emulated":
            return self._local_backend.delete(storage_key, category)
        return False

    def get_temp_dir(self) -> str:
        return self._local_backend.get_temp_dir()


# Alias for clean object naming
ObjectStorage = ObjectStorageBackend


# Singleton storage provider
_storage_backend: Optional[BaseStorageBackend] = None


def get_storage_backend() -> BaseStorageBackend:
    """Returns application storage backend singleton."""
    global _storage_backend
    if _storage_backend is None:
        _storage_backend = LocalStorageBackend()
    return _storage_backend


def get_storage() -> BaseStorageBackend:
    """Convenience alias for get_storage_backend."""
    return get_storage_backend()

