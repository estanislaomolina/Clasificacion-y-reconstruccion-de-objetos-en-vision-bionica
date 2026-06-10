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

def metric_diagnostics(percept, mask_gt, threshold_percentile=75, eps=1e-6):
    p = percept.astype(np.float32)
    gt = _resize_mask(mask_gt, p.shape)

    positive_pixels = p[p > eps]

    if positive_pixels.size == 0:
        tau = np.nan
        pred = np.zeros_like(p, dtype=bool)
    else:
        tau = np.percentile(positive_pixels, threshold_percentile)
        pred = p > tau

    return {
        "percept_min": float(p.min()),
        "percept_max": float(p.max()),
        "nonzero_pct": float((p > eps).mean()),
        "gt_area_pct": float(gt.mean()),
        "pred_area_pct": float(pred.mean()),
        "tau": float(tau) if not np.isnan(tau) else np.nan,
        "mean_obj": float(p[gt].mean()) if gt.any() else np.nan,
        "mean_bg": float(p[~gt].mean()) if (~gt).any() else np.nan,
    }


def contrast_ratio(percept: np.ndarray, mask_gt: np.ndarray, eps: float = 1e-6) -> float:
    """
    Calcula el ratio de brillo promedio entre la región del objeto
    y el fondo en el percepto simulado.

    R = mean(percept[objeto]) / mean(percept[fondo])
    """
    percept_f = percept.astype(np.float32)
    obj_mask = _resize_mask(mask_gt, percept_f.shape)
    bg_mask = ~obj_mask

    mean_obj = percept_f[obj_mask].mean() if obj_mask.any() else 0.0
    mean_bg = percept_f[bg_mask].mean() if bg_mask.any() else np.nan

    if np.isnan(mean_bg) or mean_bg < eps:
        return np.nan

    return float(mean_obj / mean_bg)

def contrast_delta(percept: np.ndarray, mask_gt: np.ndarray) -> float:
    """
    Diferencia de brillo promedio objeto - fondo.
    Más estable que el ratio cuando el fondo es casi cero.
    """
    percept_f = percept.astype(np.float32)
    obj_mask = _resize_mask(mask_gt, percept_f.shape)
    bg_mask = ~obj_mask

    mean_obj = percept_f[obj_mask].mean() if obj_mask.any() else 0.0
    mean_bg = percept_f[bg_mask].mean() if bg_mask.any() else 0.0

    return float(mean_obj - mean_bg)


def simulated_iou(
    percept: np.ndarray,
    mask_gt: np.ndarray,
    threshold_percentile: int = 75,
    eps: float = 1e-6,
) -> float:
    """
    Calcula el IoU entre la región de contraste suficiente del percepto
    y la máscara GT del objeto.

    Usa el percentil solo sobre píxeles positivos para evitar que el fondo negro
    haga colapsar el umbral a cero.
    """
    p = percept.astype(np.float32)
    gt_mask = _resize_mask(mask_gt, p.shape)

    positive_pixels = p[p > eps]

    if positive_pixels.size == 0:
        return 0.0

    tau = np.percentile(positive_pixels, threshold_percentile)

    # Usar > y no >= para evitar que tau=0 incluya todo el fondo
    pred_mask = p > tau

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

    p = percept.astype(np.float32)

    return {
        "contrast_ratio": contrast_ratio(p, mask_gt),
        "contrast_delta": contrast_delta(p, mask_gt),
        "simulated_iou":  simulated_iou(p, mask_gt, threshold_percentile),
        "ssim":           ssim_score(p, mask_gt),
    }
