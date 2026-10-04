"""
DeepVision AI — Image Preprocessing & Tensor Pipeline
Provides robust, high-performance image ingestion, color conversion,
letterbox resizing with aspect ratio preservation, coordinate unpadding,
and tensor normalization for computer vision inference.
"""

from typing import Union, Tuple, Optional
import io
import cv2
import numpy as np
from PIL import Image


def load_image(image_source: Union[str, bytes, np.ndarray, Image.Image]) -> np.ndarray:
    """
    Ingests and normalizes image sources into an RGB uint8 NumPy array (H, W, C).
    Supports file paths, raw byte streams, OpenCV BGR arrays, and PIL Images.
    """
    if isinstance(image_source, str):
        # File path
        bgr = cv2.imread(image_source)
        if bgr is None:
            raise ValueError(f"Failed to load image from file path: {image_source}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    elif isinstance(image_source, (bytes, bytearray)):
        # Byte stream from upload or network
        nparr = np.frombuffer(image_source, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Corrupt or invalid image bytes; failed to decode.")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    elif isinstance(image_source, Image.Image):
        # PIL Image
        rgb = image_source.convert("RGB")
        return np.array(rgb, dtype=np.uint8)

    elif isinstance(image_source, np.ndarray):
        # NumPy array
        if image_source.ndim == 2:
            # Grayscale -> RGB
            return cv2.cvtColor(image_source, cv2.COLOR_GRAY2RGB)
        elif image_source.ndim == 3:
            if image_source.shape[2] == 4:
                # RGBA -> RGB
                return cv2.cvtColor(image_source, cv2.COLOR_RGBA2RGB)
            elif image_source.shape[2] == 3:
                return image_source.astype(np.uint8)
        raise ValueError(f"Unsupported array shape for image: {image_source.shape}")

    else:
        raise TypeError(f"Unsupported image input type: {type(image_source)}")


def letterbox_resize(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    pad_color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resizes image maintaining original aspect ratio with symmetric letterbox padding.
    
    Args:
        image: Input RGB image (H, W, C)
        target_size: (target_width, target_height)
        pad_color: Color for padding margins
        
    Returns:
        padded_image: Resized image with letterbox borders (target_height, target_width, 3)
        scale: Scale factor applied to original image
        (pad_x, pad_y): Left and top padding offsets in pixels
    """
    orig_h, orig_w = image.shape[:2]
    target_w, target_h = target_size

    # Calculate uniform scaling factor
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    # Bilinear interpolation for downsampling / upsampling
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Calculate symmetric padding
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
        value=pad_color,
    )

    return padded, scale, (pad_x, pad_y)


def unletterbox_coordinates(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    scale: float,
    padding: Tuple[int, int],
    orig_shape: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    """
    Transforms letterboxed bounding box coordinates back to original image space.
    Clamps bounds within original image dimensions.
    """
    pad_x, pad_y = padding
    orig_h, orig_w = orig_shape

    # Invert padding and scale
    real_x_min = max(0.0, (x_min - pad_x) / scale)
    real_y_min = max(0.0, (y_min - pad_y) / scale)
    real_x_max = min(float(orig_w), (x_max - pad_x) / scale)
    real_y_max = min(float(orig_h), (y_max - pad_y) / scale)

    return (
        round(real_x_min, 2),
        round(real_y_min, 2),
        round(real_x_max, 2),
        round(real_y_max, 2),
    )


def prepare_input_tensor(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    normalize: bool = True,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Prepares a batch-ready NCHW float32 tensor from an RGB image.
    
    Returns:
        tensor: Shape (1, 3, target_h, target_w) float32 array in [0.0, 1.0]
        scale: Scale factor used in letterboxing
        padding: (pad_x, pad_y) offsets
    """
    padded, scale, padding = letterbox_resize(image, target_size=target_size)

    # Convert uint8 [0, 255] to float32 [0.0, 1.0]
    tensor = padded.astype(np.float32)
    if normalize:
        tensor /= 255.0

    # HWC -> CHW -> NCHW
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)

    return tensor, scale, padding
