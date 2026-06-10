"""
metrics.py
Métricas de evaluación cuantitativa para el pipeline del TP.

Las tres métricas principales son:
  - contrast_ratio   : ratio de brillo objeto vs fondo en el percepto simulado
  - simulated_iou    : IoU entre región de contraste suficiente y máscara GT
  - ssim_score       : SSIM entre percepto simulado y referencia ideal construida desde GT
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool)

    resized = cv2.resize(
        mask.astype(np.uint8),
        (shape[1], shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


def contrast_ratio(percept: np.ndarray, mask_gt: np.ndarray) -> float:
    """
    Calcula el ratio de brillo promedio entre la región del objeto
    y el fondo en el percepto simulado.

    R = mean(percept[objeto]) / mean(percept[fondo])

    Un R mayor indica mayor distinguibilidad.
    """
    percept_f = percept.astype(np.float32)
    obj_mask = _resize_mask(mask_gt, percept_f.shape)
    bg_mask = ~obj_mask

    mean_obj = percept_f[obj_mask].mean() if obj_mask.any() else 0.0
    mean_bg = percept_f[bg_mask].mean() if bg_mask.any() else np.nan

    if np.isnan(mean_bg) or mean_bg == 0:
        return np.nan

    return float(mean_obj / mean_bg)


def simulated_iou(
    percept: np.ndarray,
    mask_gt: np.ndarray,
    threshold_percentile: int = 75,
) -> float:
    """
    Calcula el IoU entre la región de contraste suficiente del percepto
    y la máscara GT del objeto.
    """
    tau = np.percentile(percept, threshold_percentile)
    pred_mask = (percept >= tau).astype(bool)
    gt_mask = _resize_mask(mask_gt, pred_mask.shape)

    intersection = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()

    return float(intersection / union) if union > 0 else 0.0


def ssim_score(percept: np.ndarray, mask_gt: np.ndarray) -> float:
    """
    Calcula el SSIM entre el percepto simulado y una referencia ideal
    construida a partir de la máscara GT.
    """
    p = percept.astype(np.float32)
    if p.max() > p.min():
        p = (p - p.min()) / (p.max() - p.min())

    ref = _resize_mask(mask_gt, p.shape).astype(np.float32)

    score, _ = ssim(p, ref, full=True, data_range=1.0)
    return float(score)


def compute_all_metrics(
    percept: np.ndarray,
    mask_gt: np.ndarray,
    threshold_percentile: int = 75,
) -> dict:

    # Subir el percepto a la resolución de la máscara GT
    # (más conservador que bajar la máscara, evita perder píxeles)
    if percept.shape != mask_gt.shape:
        percept_resized = cv2.resize(
            percept.astype(np.float32),
            (mask_gt.shape[1], mask_gt.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        percept_resized = percept.astype(np.float32)

    return {
        "contrast_ratio": contrast_ratio(percept_resized, mask_gt),
        "simulated_iou":  simulated_iou(percept_resized, mask_gt, threshold_percentile),
        "ssim":           ssim_score(percept_resized, mask_gt),
    }
