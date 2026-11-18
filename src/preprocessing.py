import cv2
import numpy as np


def to_original(img_rgb: np.ndarray) -> np.ndarray:
    """Baseline: skala global ke [0,1]."""
    return img_rgb.astype(np.float32) / 255.0


def to_minmax(img_rgb: np.ndarray) -> np.ndarray:
    """Min-Max per citra ke [0,1]."""
    img = img_rgb.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8:            # citra seragam, hindari bagi nol
        return np.zeros_like(img)
    return (img - mn) / (mx - mn)


def to_clahe(img_rgb: np.ndarray, clip_limit: float = 2.0,
             tile_grid: tuple = (8, 8)) -> np.ndarray:
    """CLAHE pada channel L (LAB), keluaran [0,1]."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    return out.astype(np.float32) / 255.0


PREPROCESSORS = {"original": to_original, "minmax": to_minmax, "clahe": to_clahe}