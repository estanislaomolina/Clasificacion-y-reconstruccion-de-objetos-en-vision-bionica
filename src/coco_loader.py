"""
coco_loader.py
Carga y acceso a imágenes y anotaciones de COCO 2017.
"""

from pathlib import Path
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np
from pycocotools.coco import COCO


# Clases de interés para el proyecto (subconjunto de COCO)
PRIORITY_CLASSES = {
    "person":       1,
    "car":          2,
    "bus":          2,
    "truck":        2,
    "bicycle":      2,
    "motorcycle":   2,
    "traffic light": 3,
    "stop sign":    3,
    "bench":        4,
    "chair":        4,
    "dining table": 4,
    "dog":          5,
    "cat":          5,
}


class COCOLoader:
    """
    Wrapper sobre pycocotools.COCO para el pipeline del TP.

    Parámetros
    ----------
    images_dir : str | Path
        Directorio con las imágenes (val2017/).
    annotations_path : str | Path
        Ruta al JSON de anotaciones (instances_val2017.json).
    """

    def __init__(self, images_dir: str | Path, annotations_path: str | Path):
        self.images_dir = Path(images_dir)
        self.annotations_path = Path(annotations_path)

        self.coco = COCO(str(self.annotations_path))
        self.cat_index = {cat["id"]: cat["name"] for cat in self.coco.loadCats(self.coco.getCatIds())}
        self.cat_name_to_id = {v: k for k, v in self.cat_index.items()}

    # ------------------------------------------------------------------
    # Consulta de IDs
    # ------------------------------------------------------------------

    def get_image_ids(self, cat_names: Optional[list[str]] = None) -> list[int]:
        """
        Devuelve IDs de imágenes que contienen al menos una de las clases indicadas.
        Si cat_names es None, devuelve todos los IDs del conjunto.
        """
        if cat_names is None:
            return self.coco.getImgIds()
        cat_ids = [self.cat_name_to_id[n] for n in cat_names if n in self.cat_name_to_id]
        return self.coco.getImgIds(catIds=cat_ids)

    # ------------------------------------------------------------------
    # Carga de imagen
    # ------------------------------------------------------------------

    def load_image(self, image_id: int) -> np.ndarray:
        """Carga y devuelve la imagen en BGR (OpenCV)."""
        info = self.coco.loadImgs(image_id)[0]
        path = self.images_dir / info["file_name"]
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"No se encontró la imagen: {path}")
        return img

    def load_image_rgb(self, image_id: int) -> np.ndarray:
        """Carga y devuelve la imagen en RGB."""
        return cv2.cvtColor(self.load_image(image_id), cv2.COLOR_BGR2RGB)

    def get_image_info(self, image_id: int) -> dict:
        """Devuelve el dict de metadatos de la imagen."""
        return self.coco.loadImgs(image_id)[0]

    # ------------------------------------------------------------------
    # Anotaciones y máscaras
    # ------------------------------------------------------------------

    def get_annotations(self, image_id: int) -> list[dict]:
        """Devuelve la lista de anotaciones para una imagen."""
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        return self.coco.loadAnns(ann_ids)

    def get_mask(self, ann: dict, height: int, width: int) -> np.ndarray:
        """
        Decodifica la máscara de segmentación de una anotación.
        Devuelve array binario (H, W) de dtype uint8.
        """
        return self.coco.annToMask(ann).astype(np.uint8)

    def get_instance_masks(self, image_id: int) -> list[dict]:
        """
        Devuelve una lista de dicts con:
            - 'mask'     : np.ndarray (H, W) binario
            - 'category' : str nombre de la clase
            - 'priority' : int prioridad según PRIORITY_CLASSES (0 si no está en la lista)
            - 'ann_id'   : int ID de la anotación
        """
        img_info = self.get_image_info(image_id)
        h, w = img_info["height"], img_info["width"]
        anns = self.get_annotations(image_id)

        instances = []
        for ann in anns:
            cat_name = self.cat_index.get(ann["category_id"], "unknown")
            mask = self.get_mask(ann, h, w)
            priority = PRIORITY_CLASSES.get(cat_name, 0)
            instances.append({
                "mask": mask,
                "category": cat_name,
                "priority": priority,
                "ann_id": ann["id"],
            })

        # Ordenar de menor a mayor prioridad (1 = máxima)
        instances.sort(key=lambda x: (x["priority"] if x["priority"] > 0 else 999))
        return instances
