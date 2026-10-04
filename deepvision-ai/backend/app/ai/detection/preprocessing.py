"""
DeepVision AI — Image Preprocessing & Ingestion Pipeline
Modular, production-grade image ingestion, validation, color space conversion,
letterboxing, and tensor preparation. Decoupled from specific detector architectures.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import os
import cv2
import numpy as np
from PIL import Image

from app.ai.common.types import ImageSource
from app.core.exceptions import (
    CorruptedImageException,
    EmptyImageException,
    UnsupportedImageFormatException,
    InvalidImageDimensionsException,
    ImageTooLargeException,
)

# Default safety limits to prevent decompression bomb / denial-of-service attacks
DEFAULT_MAX_DIMENSION = 8192         # Maximum width or height in pixels
DEFAULT_MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


@dataclass
class ImagePreprocessingConfig:
    """Configuration parameters for image preprocessing and tensor normalization."""
    target_size: Tuple[int, int] = (640, 640)
    normalize: bool = True
    pad_color: Tuple[int, int, int] = (114, 114, 114)  # Neutral gray
    max_dimension: int = DEFAULT_MAX_DIMENSION
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES
    to_rgb: bool = True


@dataclass
class PreprocessedImage:
    """Structured container holding prepared tensor and geometric transformation metadata."""
    tensor: np.ndarray                   # Shape (1, 3, target_h, target_w) float32
    original_image: np.ndarray           # Shape (H, W, 3) uint8 RGB
    original_shape: Tuple[int, int]      # (orig_h, orig_w)
    scale: float                         # Letterbox scaling factor
    padding: Tuple[int, int]             # (pad_x, pad_y)


class ImagePreprocessor:
    """
    Reusable, detector-agnostic image preprocessing pipeline.
    Validates inputs, handles corruptions, normalizes color channels,
    and prepares NCHW tensors with invertible coordinate mappings.
    """

    def __init__(self, config: Optional[ImagePreprocessingConfig] = None):
        self.config = config or ImagePreprocessingConfig()

    def load_image(self, source: ImageSource) -> np.ndarray:
        """
        Loads and validates an image source into an RGB uint8 NumPy array (H, W, C).
        Raises specific domain exceptions for corrupt, empty, oversized, or invalid inputs.
        """
        if source is None:
            raise EmptyImageException("Image source cannot be None.")

        # 1. Byte stream ingestion
        if isinstance(source, (bytes, bytearray)):
            if len(source) == 0:
                raise EmptyImageException("Received empty byte buffer (0 bytes).")
            if len(source) > self.config.max_file_size_bytes:
                raise ImageTooLargeException(
                    f"Image payload ({len(source)} bytes) exceeds maximum limit "
                    f"of {self.config.max_file_size_bytes} bytes."
                )

            nparr = np.frombuffer(source, np.uint8)
            img_raw = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if img_raw is None:
                try:
                    import io
                    with Image.open(io.BytesIO(source)) as pil_img:
                        rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
                except Exception:
                    raise CorruptedImageException("Corrupted or unreadable image bytes; failed to decode.")
            else:
                if img_raw.ndim == 2:
                    rgb = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2RGB)
                elif img_raw.ndim == 3:
                    if img_raw.shape[2] == 4:
                        rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGRA2RGB)
                    elif img_raw.shape[2] == 1:
                        rgb = cv2.cvtColor(img_raw[:, :, 0], cv2.COLOR_GRAY2RGB)
                    else:
                        rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
                else:
                    rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

        # 2. File path ingestion
        elif isinstance(source, str):
            if not os.path.exists(source):
                raise CorruptedImageException(f"Image file does not exist at path: {source}")
            file_size = os.path.getsize(source)
            if file_size == 0:
                raise EmptyImageException(f"Image file '{source}' is empty (0 bytes).")
            if file_size > self.config.max_file_size_bytes:
                raise ImageTooLargeException(
                    f"Image file size ({file_size} bytes) exceeds limit of "
                    f"{self.config.max_file_size_bytes} bytes."
                )

            img_raw = cv2.imread(source, cv2.IMREAD_UNCHANGED)
            if img_raw is None:
                try:
                    with Image.open(source) as pil_img:
                        rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
                except Exception:
                    raise CorruptedImageException(f"Failed to read or decode image file at: {source}")
            else:
                if img_raw.ndim == 2:
                    rgb = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2RGB)
                elif img_raw.ndim == 3:
                    if img_raw.shape[2] == 4:
                        rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGRA2RGB)
                    elif img_raw.shape[2] == 1:
                        rgb = cv2.cvtColor(img_raw[:, :, 0], cv2.COLOR_GRAY2RGB)
                    else:
                        rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
                else:
                    rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

        # 3. PIL Image ingestion
        elif isinstance(source, Image.Image):
            w, h = source.size
            if w <= 0 or h <= 0:
                raise EmptyImageException(f"Invalid PIL Image dimensions: {w}x{h}.")
            converted = source.convert("RGB")
            rgb = np.array(converted, dtype=np.uint8)

        # 4. NumPy array ingestion
        elif isinstance(source, np.ndarray):
            if source.size == 0:
                raise EmptyImageException("Input NumPy array is empty (size 0).")

            if source.ndim == 2:
                # Grayscale (H, W) -> RGB
                rgb = cv2.cvtColor(source.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            elif source.ndim == 3:
                channels = source.shape[2]
                if channels == 1:
                    rgb = cv2.cvtColor(source[:, :, 0].astype(np.uint8), cv2.COLOR_GRAY2RGB)
                elif channels == 3:
                    # Assumed RGB uint8
                    rgb = source.astype(np.uint8)
                elif channels == 4:
                    # RGBA -> RGB
                    rgb = cv2.cvtColor(source.astype(np.uint8), cv2.COLOR_RGBA2RGB)
                else:
                    raise UnsupportedImageFormatException(
                        f"Unsupported number of image channels: {channels}. Expected 1, 3, or 4."
                    )
            else:
                raise UnsupportedImageFormatException(
                    f"Unsupported NumPy array dimensions: {source.ndim}D. Expected 2D or 3D image array."
                )
        else:
            raise UnsupportedImageFormatException(
                f"Unsupported image input type: {type(source).__name__}. Expected filepath, bytes, PIL Image, or NumPy array."
            )

        # Dimension validation
        h, w = rgb.shape[:2]
        if h <= 0 or w <= 0:
            raise InvalidImageDimensionsException(f"Invalid image resolution: {w}x{h}.")
        if h > self.config.max_dimension or w > self.config.max_dimension:
            raise ImageTooLargeException(
                f"Image resolution {w}x{h} exceeds maximum allowed dimension of {self.config.max_dimension}px."
            )

        return rgb

    def resize(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Resizes image maintaining original aspect ratio with symmetric letterbox padding.
        """
        target = target_size or self.config.target_size
        orig_h, orig_w = image.shape[:2]
        target_w, target_h = target

        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        pad_right = target_w - new_w - pad_x
        pad_bottom = target_h - new_h - pad_y

        padded = cv2.copyMakeBorder(
            resized,
            top=pad_y,
            bottom=pad_bottom,
            left=pad_x,
            right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=self.config.pad_color,
        )

        return padded, scale, (pad_x, pad_y)

    def prepare_tensor(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Prepares a batch-ready NCHW float32 tensor from an RGB image.
        """
        target = target_size or self.config.target_size
        padded, scale, padding = self.resize(image, target_size=target)

        tensor = padded.astype(np.float32)
        if self.config.normalize:
            tensor /= 255.0

        # HWC -> CHW -> NCHW
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        return tensor, scale, padding

    def process(self, source: ImageSource) -> PreprocessedImage:
        """
        Executes end-to-end preprocessing pipeline:
        Validation -> Ingestion -> Color normalization -> Letterbox -> NCHW Tensor.
        """
        rgb = self.load_image(source)
        orig_h, orig_w = rgb.shape[:2]
        tensor, scale, padding = self.prepare_tensor(rgb)

        return PreprocessedImage(
            tensor=tensor,
            original_image=rgb,
            original_shape=(orig_h, orig_w),
            scale=scale,
            padding=padding,
        )

    def unletterbox_coordinates(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        scale: float,
        padding: Tuple[int, int],
        orig_shape: Tuple[int, int],
    ) -> Tuple[float, float, float, float]:
        """
        Maps letterboxed coordinates back to original image dimensions.
        """
        pad_x, pad_y = padding
        orig_h, orig_w = orig_shape

        real_x1 = max(0.0, (x1 - pad_x) / scale)
        real_y1 = max(0.0, (y1 - pad_y) / scale)
        real_x2 = min(float(orig_w), (x2 - pad_x) / scale)
        real_y2 = min(float(orig_h), (y2 - pad_y) / scale)

        return (
            round(real_x1, 2),
            round(real_y1, 2),
            round(real_x2, 2),
            round(real_y2, 2),
        )


# Global default preprocessor instance for functional API convenience
_default_preprocessor = ImagePreprocessor()


def load_image(
    image_source: ImageSource,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
) -> np.ndarray:
    """Convenience function: loads and validates image into RGB uint8 NumPy array."""
    config = ImagePreprocessingConfig(
        max_dimension=max_dimension,
        max_file_size_bytes=max_file_size_bytes,
    )
    preprocessor = ImagePreprocessor(config)
    return preprocessor.load_image(image_source)


def letterbox_resize(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    pad_color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Convenience function: letterbox resizes image preserving aspect ratio."""
    config = ImagePreprocessingConfig(target_size=target_size, pad_color=pad_color)
    preprocessor = ImagePreprocessor(config)
    return preprocessor.resize(image, target_size=target_size)


def unletterbox_coordinates(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    scale: float,
    padding: Tuple[int, int],
    orig_shape: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    """Convenience function: unpads coordinates back to original image space."""
    return _default_preprocessor.unletterbox_coordinates(
        x1=x_min,
        y1=y_min,
        x2=x_max,
        y2=y_max,
        scale=scale,
        padding=padding,
        orig_shape=orig_shape,
    )


def prepare_input_tensor(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    normalize: bool = True,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Convenience function: prepares batch NCHW float32 tensor."""
    config = ImagePreprocessingConfig(target_size=target_size, normalize=normalize)
    preprocessor = ImagePreprocessor(config)
    return preprocessor.prepare_tensor(image, target_size=target_size)
