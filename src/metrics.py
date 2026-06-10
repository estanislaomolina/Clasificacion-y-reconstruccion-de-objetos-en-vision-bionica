"""
metrics.py
Métricas de evaluación cuantitativa para el pipeline del TP.

Métricas principales recomendadas:
  - contrast_index  : contraste normalizado objeto-fondo, acotado y estable
  - contrast_delta  : diferencia absoluta de brillo objeto - fondo
  - simulated_iou   : IoU entre región de contraste suficiente y máscara GT
  - ssim_score      : SSIM entre percepto simulado y referencia ideal desde GT

Notas:
  - contrast_ratio se conserva como diagnóstico, pero no debe usarse como
    métrica principal cuando el fondo puede ser cero por construcción.
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """
    Redimensiona una máscara binaria al shape espacial del percepto.

    Parameters
    ----------
    mask : np.ndarray
        Máscara GT original, normalmente en coordenadas de la imagen COCO.
    shape : tuple[int, int]
        Shape espacial destino: (height, width).
    """
    if mask.shape == shape:
        return mask.astype(bool)

    resized = cv2.resize(
        mask.astype(np.uint8),
        (shape[1], shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


def _bbox_from_mask(mask: np.ndarray):
    """
    Devuelve bbox como (y0, y1, x0, x1), con y1/x1 exclusivos.
    Si la máscara está vacía, devuelve None.
    """
    mask = mask.astype(bool)
    if not mask.any():
        return None

    ys, xs = np.where(mask)
    return (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)


def _bbox_area(bbox) -> int:
    if bbox is None:
        return 0
    y0, y1, x0, x1 = bbox
    return max(0, y1 - y0) * max(0, x1 - x0)


def _bbox_center(bbox):
    if bbox is None:
        return (np.nan, np.nan)
    y0, y1, x0, x1 = bbox
    return ((y0 + y1) / 2.0, (x0 + x1) / 2.0)


def _bbox_iou(a, b) -> float:
    if a is None or b is None:
        return 0.0

    ay0, ay1, ax0, ax1 = a
    by0, by1, bx0, bx1 = b

    iy0 = max(ay0, by0)
    iy1 = min(ay1, by1)
    ix0 = max(ax0, bx0)
    ix1 = min(ax1, bx1)

    inter = max(0, iy1 - iy0) * max(0, ix1 - ix0)
    union = _bbox_area(a) + _bbox_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def metric_diagnostics(percept, mask_gt, threshold_percentile=75, eps=1e-6):
    """
    Diagnósticos básicos de intensidad y área para interpretar las métricas.
    """
    p = percept.astype(np.float32)
    gt = _resize_mask(mask_gt, p.shape)

    positive_pixels = p[p > eps]

    if positive_pixels.size == 0:
        tau = np.nan
        pred = np.zeros_like(p, dtype=bool)
    else:
        tau = np.percentile(positive_pixels, threshold_percentile)
        pred = p > tau

    mean_obj = float(p[gt].mean()) if gt.any() else np.nan
    mean_bg = float(p[~gt].mean()) if (~gt).any() else np.nan
    bg_near_zero = bool(np.isnan(mean_bg) or mean_bg < eps)

    return {
        "percept_min": float(p.min()),
        "percept_max": float(p.max()),
        "nonzero_pct": float((p > eps).mean()),
        "gt_area_pct": float(gt.mean()),
        "pred_area_pct": float(pred.mean()),
        "tau": float(tau) if not np.isnan(tau) else np.nan,
        "mean_obj": mean_obj,
        "mean_bg": mean_bg,
        "bg_near_zero": bg_near_zero,
        "contrast_ratio_defined": not bg_near_zero,
    }


def spatial_diagnostics(percept, mask_gt, eps=1e-6):
    """
    Diagnóstico de alineación espacial entre la zona activa del percepto y la máscara GT.

    Es útil para detectar casos como PRIMA, donde puede haber activación en el
    percepto pero esa activación cae fuera de la máscara GT redimensionada.
    """
    p = percept.astype(np.float32)
    gt = _resize_mask(mask_gt, p.shape)
    active = p > eps

    gt_bbox = _bbox_from_mask(gt)
    active_bbox = _bbox_from_mask(active)

    gt_cy, gt_cx = _bbox_center(gt_bbox)
    act_cy, act_cx = _bbox_center(active_bbox)

    h, w = p.shape
    diag = float(np.sqrt(h**2 + w**2))
    center_dist = float(np.sqrt((gt_cy - act_cy) ** 2 + (gt_cx - act_cx) ** 2))

    active_area = int(active.sum())
    gt_area = int(gt.sum())
    overlap_area = int((active & gt).sum())

    return {
        "percept_shape_h": int(h),
        "percept_shape_w": int(w),
        "gt_bbox": gt_bbox,
        "active_bbox": active_bbox,
        "gt_center_y": float(gt_cy),
        "gt_center_x": float(gt_cx),
        "active_center_y": float(act_cy),
        "active_center_x": float(act_cx),
        "bbox_iou": _bbox_iou(gt_bbox, active_bbox),
        "center_dist_norm": center_dist / diag if diag > 0 else np.nan,
        "active_in_gt_pct": float(overlap_area / active_area) if active_area > 0 else 0.0,
        "gt_covered_by_active_pct": float(overlap_area / gt_area) if gt_area > 0 else 0.0,
    }


def contrast_ratio(percept: np.ndarray, mask_gt: np.ndarray, eps: float = 1e-6) -> float:
    """
    Ratio de brillo promedio objeto/fondo.

    R = mean(percept[objeto]) / mean(percept[fondo])

    Importante: si el fondo es cero o casi cero, el ratio no está definido de
    forma informativa. En ese caso devuelve np.nan de manera explícita.
    Para comparar condiciones, usar preferentemente contrast_index y
    contrast_delta.
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


def contrast_index(percept: np.ndarray, mask_gt: np.ndarray, eps: float = 1e-6) -> float:
    """
    Contraste normalizado objeto-fondo.

    CI = (mean_obj - mean_bg) / (mean_obj + mean_bg + eps)

    Ventajas:
      - Es estable cuando el fondo es cero o casi cero.
      - Está acotado aproximadamente entre -1 y 1.
      - No recompensa artificialmente fondos negros con ratios enormes.
    """
    percept_f = percept.astype(np.float32)
    obj_mask = _resize_mask(mask_gt, percept_f.shape)
    bg_mask = ~obj_mask

    mean_obj = percept_f[obj_mask].mean() if obj_mask.any() else 0.0
    mean_bg = percept_f[bg_mask].mean() if bg_mask.any() else 0.0

    return float((mean_obj - mean_bg) / (mean_obj + mean_bg + eps))


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

    # Usar > y no >= para evitar que tau=0 incluya todo el fondo.
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
    """Calcula todas las métricas reportadas en el notebook."""
    p = percept.astype(np.float32)

    return {
        "contrast_ratio": contrast_ratio(p, mask_gt),
        "contrast_delta": contrast_delta(p, mask_gt),
        "contrast_index": contrast_index(p, mask_gt),
        "simulated_iou": simulated_iou(p, mask_gt, threshold_percentile),
        "ssim": ssim_score(p, mask_gt),
    }