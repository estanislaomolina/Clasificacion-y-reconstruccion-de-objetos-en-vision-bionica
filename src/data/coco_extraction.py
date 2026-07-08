import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import pycocotools.mask as mask_util


def letterbox(img: Image.Image, size: int) -> Image.Image:
    """
    Resize manteniendo aspect ratio y rellena con negro hasta size x size.
    """
    img = img.copy()
    img.thumbnail((size, size), Image.LANCZOS)

    padded = Image.new("L", (size, size), 0)

    offset_x = (size - img.width) // 2
    offset_y = (size - img.height) // 2

    padded.paste(img, (offset_x, offset_y))

    return padded


def decode_annotation(ann: dict, img_height: int, img_width: int):
    """
    Decodifica una segmentación COCO a una máscara binaria numpy.
    Devuelve una imagen uint8 con valores 0 o 255.
    """
    seg = ann["segmentation"]

    if isinstance(seg, list):
        rles = mask_util.frPyObjects(seg, img_height, img_width)
        rle = mask_util.merge(rles)

    elif isinstance(seg, dict):
        if isinstance(seg.get("counts"), list):
            rle = mask_util.frPyObjects(seg, img_height, img_width)
        else:
            rle = seg

    else:
        return None

    return mask_util.decode(rle).astype(np.uint8) * 255


def load_coco_annotations(annotations_file):
    """
    Carga el archivo instances_val2017.json de COCO.
    """
    annotations_file = Path(annotations_file)

    if not annotations_file.exists():
        raise FileNotFoundError(f"No existe el archivo de anotaciones: {annotations_file}")

    with open(annotations_file, "r") as f:
        coco = json.load(f)

    return coco


def get_target_category_ids(coco: dict, target_classes: list[str]):
    """
    Devuelve los category_id correspondientes a las clases objetivo.
    """
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

    target_cat_ids = {
        c["id"]
        for c in coco["categories"]
        if c["name"] in target_classes
    }

    found_classes = {cat_id_to_name[i] for i in target_cat_ids}
    missing_classes = set(target_classes) - found_classes

    return target_cat_ids, cat_id_to_name, missing_classes


def extract_coco_masks(
    annotations_file,
    output_masks_dir,
    output_metadata_file,
    target_classes,
    target_size=256,
    min_bbox_area=1000,
):
    """
    Extrae máscaras binarias desde anotaciones COCO.

    Entradas:
    - annotations_file: path al instances_val2017.json
    - output_masks_dir: carpeta donde se guardan las máscaras PNG
    - output_metadata_file: CSV final de metadata
    - target_classes: lista de clases COCO a extraer
    - target_size: tamaño final cuadrado de cada máscara
    - min_bbox_area: área mínima del bounding box

    Salidas:
    - Guarda máscaras en output_masks_dir/clase/
    - Guarda metadata en output_metadata_file
    - Devuelve un diccionario con resumen de ejecución
    """
    annotations_file = Path(annotations_file)
    output_masks_dir = Path(output_masks_dir)
    output_metadata_file = Path(output_metadata_file)

    output_masks_dir.mkdir(parents=True, exist_ok=True)
    output_metadata_file.parent.mkdir(parents=True, exist_ok=True)

    print("Cargando anotaciones COCO...")
    coco = load_coco_annotations(annotations_file)

    target_cat_ids, cat_id_to_name, missing_classes = get_target_category_ids(
        coco=coco,
        target_classes=target_classes,
    )

    if missing_classes:
        print(f"Clases no encontradas en COCO: {missing_classes}")

    img_info = {img["id"]: img for img in coco["images"]}

    for cls in target_classes:
        class_dir = cls.replace(" ", "_")
        (output_masks_dir / class_dir).mkdir(parents=True, exist_ok=True)

    annotations = [
        ann
        for ann in coco["annotations"]
        if ann["category_id"] in target_cat_ids
    ]

    print(f"Anotaciones encontradas para las clases objetivo: {len(annotations)}")

    metadata_rows = []
    saved = 0
    skipped = 0

    for ann in annotations:
        x, y, w, h = [int(v) for v in ann["bbox"]]

        if w * h < min_bbox_area:
            skipped += 1
            continue

        img_meta = img_info[ann["image_id"]]
        img_h = img_meta["height"]
        img_w = img_meta["width"]

        class_name = cat_id_to_name[ann["category_id"]]
        class_dir = class_name.replace(" ", "_")

        mask = decode_annotation(ann, img_h, img_w)

        if mask is None:
            skipped += 1
            continue

        x2 = min(x + w, img_w)
        y2 = min(y + h, img_h)

        crop = mask[y:y2, x:x2]

        if crop.size == 0:
            skipped += 1
            continue

        crop_img = Image.fromarray(crop, mode="L")
        resized = letterbox(crop_img, target_size)

        sample_id = f"{ann['image_id']}_{ann['id']}"
        filename = f"{sample_id}.png"

        save_path = output_masks_dir / class_dir / filename
        resized.save(save_path)

        metadata_rows.append({
            "sample_id": sample_id,
            "filename": str(save_path),
            "class": class_name,
            "image_id": ann["image_id"],
            "ann_id": ann["id"],
            "orig_img_w": img_w,
            "orig_img_h": img_h,
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": w,
            "bbox_h": h,
            "bbox_area": w * h,
        })

        saved += 1

        if saved % 500 == 0:
            print(f"{saved} máscaras guardadas...")

    if not metadata_rows:
        raise RuntimeError(
            "No se guardó ninguna máscara. Revisá target_classes, annotations_file o min_bbox_area."
        )

    with open(output_metadata_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metadata_rows[0].keys())
        writer.writeheader()
        writer.writerows(metadata_rows)

    counts = Counter(row["class"] for row in metadata_rows)

    summary = {
        "saved": saved,
        "skipped": skipped,
        "metadata_file": str(output_metadata_file),
        "output_masks_dir": str(output_masks_dir),
        "class_counts": dict(counts),
        "missing_classes": sorted(list(missing_classes)),
    }

    return summary


def print_extraction_summary(summary: dict):
    """
    Imprime un resumen limpio de la extracción.
    """
    print("\nListo.")
    print(f"Guardadas: {summary['saved']}")
    print(f"Descartadas: {summary['skipped']}")
    print(f"Metadata: {summary['metadata_file']}")
    print(f"Masks dir: {summary['output_masks_dir']}")

    print("\nDistribución por clase:")
    for cls, n in sorted(summary["class_counts"].items()):
        print(f"{cls:<20} {n}")

    if summary["missing_classes"]:
        print("\nClases faltantes:")
        for cls in summary["missing_classes"]:
            print("-", cls)
