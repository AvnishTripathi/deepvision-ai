"""
DeepVision AI — Base Identity Verification Interface
Prepares architectural contract for future consent-based authorized identity verification.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np


class BaseIdentityVerifier(ABC):
    """
    Abstract interface for consent-based authorized identity verification.
    Strictly follows security, privacy, and consent compliance.
    """

    @abstractmethod
    def extract_embedding(self, aligned_face_crop: np.ndarray) -> np.ndarray:
        """Extracts normalized L2 facial embedding vector from aligned face crop."""
        pass

    @abstractmethod
    def verify_identity(self, source_embedding: np.ndarray, target_embedding: np.ndarray, threshold: float = 0.6) -> bool:
        """Computes cosine similarity between authorized embeddings."""
        pass
