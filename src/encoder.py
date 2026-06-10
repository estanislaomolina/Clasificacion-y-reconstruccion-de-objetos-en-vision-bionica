"""
encoder.py
Encoder semántico: toma una imagen BGR y una lista de instancias con máscaras GT,
y produce una imagen modificada donde los objetos prioritarios están resaltados
y el fondo está atenuado.
"""

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np


EnhanceTechnique = Literal["brightness", "contrast", "edges_canny", "edges_sobel", "dilate"]
AttenuateTechnique = Literal["dim", "blur", "black"]


@dataclass
class EncoderConfig:
    """
    Parámetros configurables del encoder semántico.
    Modificar estos valores es el espacio de experimentación principal del TP.
    """
    # Técnicas de resalte aplicadas a objetos de alta prioridad
    enhance_techniques: list[EnhanceTechnique] = field(
        default_factory=lambda: ["brightness", "edges_canny"]
    )
    # Factor de incremento de brillo (0.0 = sin cambio, 1.0 = doble brillo)
    brightness_factor: float = 0.5
    # Factor de incremento de contraste (1.0 = sin cambio)
    contrast_factor: float = 1.4
    # Umbral inferior para Canny
    canny_threshold1: int = 50
    # Umbral superior para Canny
    canny_threshold2: int = 150
    # Kernel de dilatación de máscaras (píxeles)
    dilation_kernel_size: int = 5
    # Técnica de atenuación del fondo
    attenuate_technique: AttenuateTechnique = "dim"
    # Factor de atenuación del fondo (0.0 = negro, 1.0 = sin cambio)
    dim_factor: float = 0.2
    # Radio del blur gaussiano sobre el fondo (debe ser impar)
    blur_kernel_size: int = 21
    # Prioridad mínima para considerar un objeto como "relevante"
    # Objetos con priority > min_priority o priority == 0 se tratan como fondo
    min_priority: int = 3


class SemanticEncoder:
    """
    Aplica el encoder semántico sobre una imagen usando máscaras GT de COCO.

    Uso
    ---
    encoder = SemanticEncoder(config)
    result = encoder.encode(image_bgr, instances)
    """

    def __init__(self, config: EncoderConfig = None):
        self.config = config or EncoderConfig()

    # ------------------------------------------------------------------
    # Método principal
    # ------------------------------------------------------------------

    def encode(self, image: np.ndarray, instances: list[dict]) -> np.ndarray:
        """
        Parámetros
        ----------
        image : np.ndarray
            Imagen en BGR, shape (H, W, 3).
        instances : list[dict]
            Lista devuelta por COCOLoader.get_instance_masks().

        Devuelve
        --------
        np.ndarray
            Imagen modificada en BGR, misma shape que la entrada.
        """
        result = image.copy().astype(np.float32)
        h, w = image.shape[:2]

        # Máscara acumulada de todos los objetos relevantes
        foreground_mask = np.zeros((h, w), dtype=np.uint8)

        for inst in instances:
            if inst["priority"] == 0 or inst["priority"] > self.config.min_priority:
                continue
            mask = inst["mask"]
            if self.config.dilation_kernel_size > 1:
                kernel = np.ones(
                    (self.config.dilation_kernel_size, self.config.dilation_kernel_size),
                    dtype=np.uint8
                )
                mask = cv2.dilate(mask, kernel, iterations=1)
            foreground_mask = np.maximum(foreground_mask, mask)
            result = self._enhance_region(result, mask)

        # Atenuar fondo
        background_mask = 1 - foreground_mask
        result = self._attenuate_background(result, image.astype(np.float32), background_mask)

        return np.clip(result, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Resalte de objetos prioritarios
    # ------------------------------------------------------------------

    def _enhance_region(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        result = image.copy()
        bool_mask = mask.astype(bool)

        for technique in self.config.enhance_techniques:
            if technique == "brightness":
                result[bool_mask] = np.clip(
                    result[bool_mask] * (1 + self.config.brightness_factor), 0, 255
                )
            elif technique == "contrast":
                mean = result[bool_mask].mean()
                result[bool_mask] = np.clip(
                    mean + (result[bool_mask] - mean) * self.config.contrast_factor, 0, 255
                )
            elif technique == "edges_canny":
                gray = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, self.config.canny_threshold1, self.config.canny_threshold2)
                edges_masked = edges * mask
                edge_pixels = edges_masked.astype(bool)
                result[edge_pixels] = 255
            elif technique == "edges_sobel":
                gray = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                sobel = np.sqrt(sx**2 + sy**2)
                sobel = (sobel / sobel.max() * 255).astype(np.uint8) if sobel.max() > 0 else sobel.astype(np.uint8)
                sobel_masked = sobel * mask
                edge_pixels = sobel_masked > 30
                result[edge_pixels] = 255

        return result

    # ------------------------------------------------------------------
    # Atenuación del fondo
    # ------------------------------------------------------------------

    def _attenuate_background(
        self,
        result: np.ndarray,
        original: np.ndarray,
        background_mask: np.ndarray,
    ) -> np.ndarray:
        bool_mask = background_mask.astype(bool)

        if self.config.attenuate_technique == "dim":
            result[bool_mask] = original[bool_mask] * self.config.dim_factor
        elif self.config.attenuate_technique == "blur":
            blurred = cv2.GaussianBlur(
                original.astype(np.uint8),
                (self.config.blur_kernel_size, self.config.blur_kernel_size),
                0
            ).astype(np.float32)
            result[bool_mask] = blurred[bool_mask]
        elif self.config.attenuate_technique == "black":
            result[bool_mask] = 0

        return result
