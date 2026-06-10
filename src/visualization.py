"""
visualization.py
Funciones de visualización para el pipeline del TP.
Genera figuras comparativas entre las tres condiciones experimentales.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def overlay_masks(image_rgb: np.ndarray, instances: list[dict], alpha: float = 0.4) -> np.ndarray:
    """
    Dibuja las máscaras GT sobre la imagen con colores por prioridad.

    Colores:
        prioridad 1 → rojo
        prioridad 2 → naranja
        prioridad 3 → amarillo
        prioridad 4+ → verde
        sin prioridad → gris
    """
    PRIORITY_COLORS = {
        1: (255, 60,  60),
        2: (255, 165,  0),
        3: (255, 230,  0),
        4: (100, 200, 100),
        5: (100, 200, 100),
    }
    DEFAULT_COLOR = (180, 180, 180)

    overlay = image_rgb.copy().astype(np.float32)

    for inst in instances:
        mask = inst["mask"].astype(bool)
        color = PRIORITY_COLORS.get(inst["priority"], DEFAULT_COLOR)
        for c, val in enumerate(color):
            overlay[:, :, c][mask] = (
                overlay[:, :, c][mask] * (1 - alpha) + val * alpha
            )

    # Leyenda de clases detectadas
    result = np.clip(overlay, 0, 255).astype(np.uint8)
    return result


def plot_comparison(
    image_rgb: np.ndarray,
    encoded_bgr: np.ndarray,
    percept_baseline: np.ndarray,
    percept_edges: np.ndarray,
    percept_semantic: np.ndarray,
    implant_name: str = "Argus II",
    title: str = "",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Genera la figura comparativa de las cinco vistas:
        1. Imagen original
        2. Imagen tras el encoder semántico
        3. Percepto baseline
        4. Percepto edge-enhanced
        5. Percepto semántico

    Parámetros
    ----------
    image_rgb : np.ndarray
        Imagen original en RGB.
    encoded_bgr : np.ndarray
        Imagen modificada por el encoder semántico (BGR).
    percept_baseline : np.ndarray
        Output de pulse2percept sobre imagen original.
    percept_edges : np.ndarray
        Output de pulse2percept sobre imagen con bordes resaltados.
    percept_semantic : np.ndarray
        Output de pulse2percept sobre imagen codificada semánticamente.
    implant_name : str
        Nombre del implante simulado (para el título).
    save_path : str | Path, optional
        Si se indica, guarda la figura en esa ruta.
    """
    encoded_rgb = cv2.cvtColor(encoded_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(
        f"{title}  —  Implante: {implant_name}" if title else f"Implante: {implant_name}",
        fontsize=13, fontweight="bold", y=1.02
    )

    panels = [
        (image_rgb,        "Original",                "viridis"),
        (encoded_rgb,      "Encoder semántico",       "viridis"),
        (percept_baseline, "Baseline (sin procesar)", "gray"),
        (percept_edges,    "Edge-enhanced",           "gray"),
        (percept_semantic, "Semántico",               "gray"),
    ]

    for ax, (img, label, cmap) in zip(axes, panels):
        if img.ndim == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap)
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_metrics_by_condition(
    metrics_df,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Genera un gráfico de barras comparando las tres métricas
    por condición y por implante.

    Parámetros
    ----------
    metrics_df : pd.DataFrame
        DataFrame con columnas: condition, implant, contrast_ratio, simulated_iou, ssim.
    save_path : str | Path, optional
        Ruta para guardar la figura.
    """
    import pandas as pd

    metric_cols = ["contrast_ratio", "simulated_iou", "ssim"]
    metric_labels = ["Contraste objeto/fondo", "IoU simulado", "SSIM"]

    conditions = metrics_df["condition"].unique()
    implants = metrics_df["implant"].unique()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Comparación de métricas por condición e implante", fontsize=13, fontweight="bold")

    colors = plt.cm.tab10.colors

    for ax, col, label in zip(axes, metric_cols, metric_labels):
        x = np.arange(len(conditions))
        width = 0.25
        for i, implant in enumerate(implants):
            vals = [
                metrics_df[(metrics_df["condition"] == c) & (metrics_df["implant"] == implant)][col].mean()
                for c in conditions
            ]
            ax.bar(x + i * width, vals, width, label=implant, color=colors[i])

        ax.set_title(label, fontsize=11)
        ax.set_xticks(x + width)
        ax.set_xticklabels(conditions, rotation=15, ha="right", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
