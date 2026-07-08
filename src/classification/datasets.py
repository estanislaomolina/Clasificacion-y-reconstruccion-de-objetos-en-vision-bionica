from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def create_sample_split(
    df,
    test_size=0.15,
    val_size=0.15,
    seed=42,
    stratify=True,
):
    """
    Crea split train / val / test por sample_id.

    Esto evita que perceptos de un mismo objeto original caigan
    en splits distintos.

    df debe tener:
    - sample_id
    - label
    """
    required_cols = {"sample_id", "label"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Faltan columnas en metadata: {missing}")

    sample_df = (
        df[["sample_id", "label"]]
        .drop_duplicates("sample_id")
        .reset_index(drop=True)
    )

    tmp_size = val_size + test_size

    if tmp_size <= 0 or tmp_size >= 1:
        raise ValueError("val_size + test_size debe estar entre 0 y 1.")

    stratify_labels = sample_df["label"] if stratify else None

    try:
        train_ids, tmp_ids = train_test_split(
            sample_df["sample_id"],
            test_size=tmp_size,
            random_state=seed,
            stratify=stratify_labels,
        )
    except ValueError:
        print("Aviso: no se pudo hacer split estratificado. Uso split simple.")
        train_ids, tmp_ids = train_test_split(
            sample_df["sample_id"],
            test_size=tmp_size,
            random_state=seed,
            stratify=None,
        )

    tmp_df = sample_df[sample_df["sample_id"].isin(tmp_ids)].copy()

    relative_test_size = test_size / tmp_size

    stratify_tmp = tmp_df["label"] if stratify else None

    try:
        val_ids, test_ids = train_test_split(
            tmp_df["sample_id"],
            test_size=relative_test_size,
            random_state=seed,
            stratify=stratify_tmp,
        )
    except ValueError:
        print("Aviso: no se pudo hacer val/test estratificado. Uso split simple.")
        val_ids, test_ids = train_test_split(
            tmp_df["sample_id"],
            test_size=relative_test_size,
            random_state=seed,
            stratify=None,
        )

    df_train = df[df["sample_id"].isin(train_ids)].copy()
    df_val = df[df["sample_id"].isin(val_ids)].copy()
    df_test = df[df["sample_id"].isin(test_ids)].copy()

    return df_train, df_val, df_test


def check_split_leakage(df_train, df_val, df_test):
    """
    Verifica que no haya sample_id compartidos entre splits.
    """
    train_ids = set(df_train["sample_id"])
    val_ids = set(df_val["sample_id"])
    test_ids = set(df_test["sample_id"])

    train_val = train_ids & val_ids
    train_test = train_ids & test_ids
    val_test = val_ids & test_ids

    print("Train ∩ Val:", len(train_val))
    print("Train ∩ Test:", len(train_test))
    print("Val ∩ Test:", len(val_test))

    if train_val or train_test or val_test:
        raise RuntimeError("Hay leakage entre splits por sample_id.")

    print("Split OK: no hay leakage por sample_id.")


def load_grayscale_image(path):
    """
    Carga una imagen en escala de grises.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    return Image.open(path).convert("L")


def pil_to_tensor(img):
    """
    Convierte PIL grayscale a tensor [1, H, W] en rango [0, 1].
    Se usa si no se pasa transform.
    """
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0)
    return tensor


class PerceptDataset(Dataset):
    """
    Dataset para entrenar con un percepto individual.

    Cada fila representa:
    - sample_id
    - label
    - implant
    - percept_path
    """

    def __init__(self, df, classes, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.classes = list(classes)
        self.cls2idx = {cls: i for i, cls in enumerate(self.classes)}
        self.transform = transform

        required_cols = {"label", "percept_path"}
        missing = required_cols - set(self.df.columns)

        if missing:
            raise ValueError(f"Faltan columnas en df: {missing}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = load_grayscale_image(row["percept_path"])

        if self.transform is not None:
            x = self.transform(img)
        else:
            x = pil_to_tensor(img)

        y = self.cls2idx[row["label"]]
        y = torch.tensor(y, dtype=torch.long)

        return x, y


class PerceptSeqDataset(Dataset):
    """
    Dataset multimplante.

    Para cada sample_id construye una secuencia:

    [
        percepto_argus2,
        percepto_prima,
        percepto_alphams
    ]

    Devuelve:
    - x: tensor [T, 1, H, W]
    - y: label
    """

    def __init__(self, df, implants, classes, transform=None):
        self.df = df.copy()
        self.implants = list(implants)
        self.classes = list(classes)
        self.cls2idx = {cls: i for i, cls in enumerate(self.classes)}
        self.transform = transform

        required_cols = {"sample_id", "label", "implant", "percept_path"}
        missing = required_cols - set(self.df.columns)

        if missing:
            raise ValueError(f"Faltan columnas en df: {missing}")

        pivot = (
            self.df.pivot_table(
                index="sample_id",
                columns="implant",
                values="percept_path",
                aggfunc="first",
            )
            .dropna(subset=self.implants)
            .reset_index()
        )

        label_map = (
            self.df.drop_duplicates("sample_id")
            .set_index("sample_id")["label"]
        )

        pivot["label"] = pivot["sample_id"].map(label_map)

        self.samples = pivot.reset_index(drop=True)

        if self.samples.empty:
            raise ValueError(
                "PerceptSeqDataset quedó vacío. "
                "Revisá que cada sample_id tenga todos los implantes."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]

        frames = []

        for implant in self.implants:
            img = load_grayscale_image(row[implant])

            if self.transform is not None:
                x = self.transform(img)
            else:
                x = pil_to_tensor(img)

            frames.append(x)

        x_seq = torch.stack(frames, dim=0)

        y = self.cls2idx[row["label"]]
        y = torch.tensor(y, dtype=torch.long)

        return x_seq, y