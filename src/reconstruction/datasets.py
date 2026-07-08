from pathlib import Path
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class JointTransform:
    """
    Aplica la MISMA transformación geométrica (flip, rotación) a percepto
    y máscara a la vez -- si no, se desalinean. Normalize solo va al percepto.
    """

    def __init__(self, size, train=True, hflip_p=0.5, max_rotation=10):
        self.size = size
        self.train = train
        self.hflip_p = hflip_p
        self.max_rotation = max_rotation

    def __call__(self, percept_img, mask_img):
        percept_img = percept_img.resize((self.size, self.size), Image.BILINEAR)
        mask_img = mask_img.resize((self.size, self.size), Image.NEAREST)

        if self.train:
            if random.random() < self.hflip_p:
                percept_img = F.hflip(percept_img)
                mask_img = F.hflip(mask_img)
            angle = random.uniform(-self.max_rotation, self.max_rotation)
            percept_img = F.rotate(percept_img, angle, fill=0)
            mask_img = F.rotate(mask_img, angle, fill=0)

        percept_t = F.to_tensor(percept_img)
        percept_t = F.normalize(percept_t, mean=[0.5], std=[0.5])

        mask_t = F.to_tensor(mask_img)
        mask_t = (mask_t > 0.5).float()

        return percept_t, mask_t


class PerceptReconstructionDataset(Dataset):
    """
    Pares (percepto, máscara) para UN implante.
    Esperá un df ya filtrado por implante (mismo patrón que PerceptDataset).
    """

    def __init__(self, df, img_size=256, train=False):
        self.df = df.reset_index(drop=True)
        self.transform = JointTransform(size=img_size, train=train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        percept_img = Image.open(row["percept_path"]).convert("L")
        mask_img = Image.open(row["mask_path"]).convert("L")
        return self.transform(percept_img, mask_img)


class PerceptFusionDataset(Dataset):
    """
    Para cada sample_id, devuelve los percepts de TODOS los implantes
    apilados como canales (n_implants, H, W) + la máscara compartida.
    """

    def __init__(self, df, implants, img_size=256, train=False):
        self.implants = implants
        self.transform = JointTransform(size=img_size, train=train)

        rows = []
        for sample_id, group in df.groupby("sample_id"):
            group = group.set_index("implant")
            if not all(imp in group.index for imp in implants):
                continue
            rows.append({
                "sample_id": sample_id,
                "mask_path": group["mask_path"].iloc[0],
                **{f"percept_path_{imp}": group.loc[imp, "percept_path"] for imp in implants},
            })
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        mask_img = Image.open(row["mask_path"]).convert("L")

        # misma geometría (flip/rotación) para los 3 implantes + la máscara
        seed = random.randint(0, 10_000_000)
        percept_tensors = []
        mask_t = None
        for imp in self.implants:
            percept_img = Image.open(row[f"percept_path_{imp}"]).convert("L")
            random.seed(seed)
            percept_t, mask_t = self.transform(percept_img, mask_img)
            percept_tensors.append(percept_t)

        percepts = torch.cat(percept_tensors, dim=0)  # (n_implants, H, W)
        return percepts, mask_t
    
# Nuevo para reconstrucción de imagen RGB a partir de perceptos

class PerceptToImageDataset(Dataset):
    """
    Pares (percepto, crop RGB de COCO) para UN implante.
    Necesita df con columnas: percept_path, coco_image_path,
    bbox_x, bbox_y, bbox_w, bbox_h.
    """

    def __init__(self, df, img_size=256, train=False):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- percepto (grayscale) ---
        percept_img = Image.open(row["percept_path"]).convert("L")

        # --- crop RGB de COCO ---
        coco_img = Image.open(row["coco_image_path"]).convert("RGB")
        x, y, w, h = int(row["bbox_x"]), int(row["bbox_y"]), int(row["bbox_w"]), int(row["bbox_h"])
        # clamp por si el bbox se sale de los bordes
        iw, ih = coco_img.size
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(iw, x + w)
        y2 = min(ih, y + h)
        target_img = coco_img.crop((x1, y1, x2, y2))

        # --- misma geometría para percepto y target ---
        percept_img = percept_img.resize((self.img_size, self.img_size), Image.BILINEAR)
        target_img = target_img.resize((self.img_size, self.img_size), Image.BILINEAR)

        if self.train:
            if random.random() < 0.5:
                percept_img = F.hflip(percept_img)
                target_img = F.hflip(target_img)
            angle = random.uniform(-10, 10)
            percept_img = F.rotate(percept_img, angle, fill=0)
            target_img = F.rotate(target_img, angle, fill=0)

        # percepto: normalizar a [-1, 1]
        percept_t = F.to_tensor(percept_img)           # [1, H, W]
        percept_t = F.normalize(percept_t, mean=[0.5], std=[0.5])

        # target RGB: normalizar a [-1, 1]
        target_t = F.to_tensor(target_img)             # [3, H, W]
        target_t = F.normalize(target_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        return percept_t, target_t


class PerceptFusionToImageDataset(Dataset):
    """
    Para cada sample_id, apila los percepts de todos los implantes
    como canales (n_implants, H, W) y devuelve el crop RGB como target.
    """

    def __init__(self, df, implants, img_size=256, train=False):
        self.implants = implants
        self.img_size = img_size
        self.train = train

        rows = []
        for sample_id, group in df.groupby("sample_id"):
            group_idx = group.set_index("implant")
            if not all(imp in group_idx.index for imp in implants):
                continue
            first = group_idx.iloc[0]
            rows.append({
                "sample_id": sample_id,
                "coco_image_path": first["coco_image_path"],
                "bbox_x": first["bbox_x"],
                "bbox_y": first["bbox_y"],
                "bbox_w": first["bbox_w"],
                "bbox_h": first["bbox_h"],
                **{f"percept_path_{imp}": group_idx.loc[imp, "percept_path"] for imp in implants},
            })
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        # crop RGB de COCO
        coco_img = Image.open(row["coco_image_path"]).convert("RGB")
        x, y, w, h = int(row["bbox_x"]), int(row["bbox_y"]), int(row["bbox_w"]), int(row["bbox_h"])
        iw, ih = coco_img.size
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(iw, x + w), min(ih, y + h)
        target_img = coco_img.crop((x1, y1, x2, y2))
        target_img = target_img.resize((self.img_size, self.img_size), Image.BILINEAR)

        # misma transformación geométrica para todos los percepts y el target
        seed = random.randint(0, 10_000_000)
        percept_tensors = []
        for imp in self.implants:
            percept_img = Image.open(row[f"percept_path_{imp}"]).convert("L")
            percept_img = percept_img.resize((self.img_size, self.img_size), Image.BILINEAR)
            random.seed(seed)
            if self.train:
                if random.random() < 0.5:
                    percept_img = F.hflip(percept_img)
                angle = random.uniform(-10, 10)
                percept_img = F.rotate(percept_img, angle, fill=0)
            t = F.to_tensor(percept_img)
            t = F.normalize(t, mean=[0.5], std=[0.5])
            percept_tensors.append(t)

        # aplicar misma geometría al target
        random.seed(seed)
        if self.train:
            if random.random() < 0.5:
                target_img = F.hflip(target_img)
            angle = random.uniform(-10, 10)
            target_img = F.rotate(target_img, angle, fill=0)

        percepts = torch.cat(percept_tensors, dim=0)  # (n_implants, H, W)
        target_t = F.to_tensor(target_img)
        target_t = F.normalize(target_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        return percepts, target_t