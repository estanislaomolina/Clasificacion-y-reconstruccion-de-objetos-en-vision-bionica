from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
from IPython.display import display

from src.classification.knn_baselines import load_percept_vector


def _load_image(path, mode="RGB"):
    path = Path(path)

    if not path.exists():
        return None

    return Image.open(path).convert(mode)


def _get_coco_original_path(row, coco_images_dir):
    """
    Reconstruye el path de la imagen original COCO.

    COCO val2017 usa nombres:
    000000000139.jpg
    """
    image_id = int(row["image_id"])
    filename = f"{image_id:012d}.jpg"

    return Path(coco_images_dir) / filename


def _format_confidence(confidence):
    if confidence is None or np.isnan(confidence):
        return "NA"

    return f"{confidence:.3f}"


def _predict_sklearn_model(model, x, classes):
    """
    Predice con un modelo sklearn/pipeline y devuelve:
    - pred_idx
    - pred_label
    - confidence
    - full_probs
    """
    pred_idx = int(model.predict(x)[0])

    full_probs = np.full(len(classes), np.nan, dtype=float)
    confidence = np.nan

    if hasattr(model, "predict_proba"):
        raw_probs = model.predict_proba(x)[0]

        class_order = getattr(
            model,
            "classes_",
            np.arange(len(raw_probs)),
        )

        full_probs = np.zeros(len(classes), dtype=float)

        for cls_idx, prob in zip(class_order, raw_probs):
            cls_idx = int(cls_idx)

            if 0 <= cls_idx < len(classes):
                full_probs[cls_idx] = float(prob)

        confidence = float(full_probs[pred_idx])

    return pred_idx, classes[pred_idx], confidence, full_probs


def _predict_knn_single(
    sample_rows,
    knn_results,
    implant,
    classes,
    knn_img_size,
):
    key = f"knn_{implant}"

    if key not in knn_results:
        return None

    row_imp = sample_rows[sample_rows["implant"] == implant]

    if row_imp.empty:
        return None

    row = row_imp.iloc[0]

    x = load_percept_vector(
        row["percept_path"],
        img_size=knn_img_size,
    ).reshape(1, -1)

    model = knn_results[key]["model"]

    pred_idx, pred_label, confidence, probs = _predict_sklearn_model(
        model=model,
        x=x,
        classes=classes,
    )

    return {
        "model": f"KNN {implant}",
        "family": "KNN single",
        "implant": implant,
        "prediction": pred_label,
        "confidence": confidence,
        "probs": probs,
    }


def _predict_knn_multi(
    sample_rows,
    knn_results,
    implants,
    classes,
    knn_img_size,
):
    key = "knn_multi_implant"

    if key not in knn_results:
        return None

    features = []

    for implant in implants:
        row_imp = sample_rows[sample_rows["implant"] == implant]

        if row_imp.empty:
            return None

        row = row_imp.iloc[0]

        vec = load_percept_vector(
            row["percept_path"],
            img_size=knn_img_size,
        )

        features.append(vec)

    x = np.concatenate(features).reshape(1, -1)

    model = knn_results[key]["model"]

    pred_idx, pred_label, confidence, probs = _predict_sklearn_model(
        model=model,
        x=x,
        classes=classes,
    )

    return {
        "model": "KNN multi-implant",
        "family": "KNN multi",
        "implant": "all",
        "prediction": pred_label,
        "confidence": confidence,
        "probs": probs,
    }


@torch.no_grad()
def _predict_cnn_single(
    sample_rows,
    cnn_models,
    implant,
    classes,
    transform,
    device,
):
    if implant not in cnn_models:
        return None

    row_imp = sample_rows[sample_rows["implant"] == implant]

    if row_imp.empty:
        return None

    row = row_imp.iloc[0]

    img = Image.open(row["percept_path"]).convert("L")
    x = transform(img).unsqueeze(0).to(device)

    model = cnn_models[implant]
    model.eval()

    logits = model(x)
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_label = classes[pred_idx]
    confidence = float(probs[pred_idx])

    return {
        "model": f"CNN {implant}",
        "family": "CNN single",
        "implant": implant,
        "prediction": pred_label,
        "confidence": confidence,
        "probs": probs,
    }


@torch.no_grad()
def _predict_cnn_gru(
    sample_rows,
    cnn_gru_model,
    implants,
    classes,
    transform,
    device,
):
    if cnn_gru_model is None:
        return None

    frames = []

    for implant in implants:
        row_imp = sample_rows[sample_rows["implant"] == implant]

        if row_imp.empty:
            return None

        row = row_imp.iloc[0]

        img = Image.open(row["percept_path"]).convert("L")
        x = transform(img)

        frames.append(x)

    x_seq = torch.stack(frames, dim=0)
    x_seq = x_seq.unsqueeze(0).to(device)

    cnn_gru_model.eval()

    logits = cnn_gru_model(x_seq)
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_label = classes[pred_idx]
    confidence = float(probs[pred_idx])

    return {
        "model": "CNN + GRU",
        "family": "CNN multi",
        "implant": "all",
        "prediction": pred_label,
        "confidence": confidence,
        "probs": probs,
    }


def _collect_predictions(
    sample_rows,
    classes,
    implants,
    knn_results=None,
    cnn_models=None,
    cnn_gru_model=None,
    transform=None,
    device="cuda",
    knn_img_size=64,
):
    predictions = []

    if knn_results is not None:
        for implant in implants:
            pred = _predict_knn_single(
                sample_rows=sample_rows,
                knn_results=knn_results,
                implant=implant,
                classes=classes,
                knn_img_size=knn_img_size,
            )

            if pred is not None:
                predictions.append(pred)

        pred_multi = _predict_knn_multi(
            sample_rows=sample_rows,
            knn_results=knn_results,
            implants=implants,
            classes=classes,
            knn_img_size=knn_img_size,
        )

        if pred_multi is not None:
            predictions.append(pred_multi)

    if cnn_models is not None and transform is not None:
        for implant in implants:
            pred = _predict_cnn_single(
                sample_rows=sample_rows,
                cnn_models=cnn_models,
                implant=implant,
                classes=classes,
                transform=transform,
                device=device,
            )

            if pred is not None:
                predictions.append(pred)

    if cnn_gru_model is not None and transform is not None:
        pred_gru = _predict_cnn_gru(
            sample_rows=sample_rows,
            cnn_gru_model=cnn_gru_model,
            implants=implants,
            classes=classes,
            transform=transform,
            device=device,
        )

        if pred_gru is not None:
            predictions.append(pred_gru)

    return predictions


def _predictions_to_dataframe(predictions, classes, true_label):
    rows = []

    for pred in predictions:
        row = {
            "model": pred["model"],
            "family": pred["family"],
            "implant": pred["implant"],
            "true_label": true_label,
            "prediction": pred["prediction"],
            "correct": pred["prediction"] == true_label,
            "confidence": pred["confidence"],
        }

        probs = pred.get("probs")

        if probs is not None:
            for i, cls in enumerate(classes):
                row[f"prob_{cls}"] = float(probs[i])

        rows.append(row)

    return pd.DataFrame(rows)


def _plot_images(
    sample_rows,
    sample_id,
    classes,
    implants,
    coco_images_dir,
    save_path=None,
):
    first_row = sample_rows.iloc[0]

    true_label = first_row["label"]
    image_id = first_row["image_id"]
    ann_id = first_row["ann_id"]

    original_path = _get_coco_original_path(
        row=first_row,
        coco_images_dir=coco_images_dir,
    )

    mask_path = Path(first_row["mask_path"])

    original_img = _load_image(original_path, mode="RGB")
    mask_img = _load_image(mask_path, mode="L")

    n_cols = 2 + len(implants)

    fig = plt.figure(figsize=(4 * n_cols, 4))

    ax = plt.subplot(1, n_cols, 1)

    if original_img is not None:
        ax.imshow(original_img)
    else:
        ax.text(
            0.5,
            0.5,
            "Imagen original\nno encontrada",
            ha="center",
            va="center",
        )

    ax.set_title("Imagen original")
    ax.axis("off")

    ax = plt.subplot(1, n_cols, 2)

    if mask_img is not None:
        ax.imshow(mask_img, cmap="gray")
    else:
        ax.text(
            0.5,
            0.5,
            "Máscara\nno encontrada",
            ha="center",
            va="center",
        )

    ax.set_title(f"Máscara\ntrue: {true_label}")
    ax.axis("off")

    for col_idx, implant in enumerate(implants, start=3):
        ax = plt.subplot(1, n_cols, col_idx)

        row_imp = sample_rows[sample_rows["implant"] == implant]

        if row_imp.empty:
            ax.text(
                0.5,
                0.5,
                f"{implant}\nno disponible",
                ha="center",
                va="center",
            )
            ax.set_title(implant)
            ax.axis("off")
            continue

        percept_path = Path(row_imp.iloc[0]["percept_path"])
        percept_img = _load_image(percept_path, mode="L")

        if percept_img is not None:
            ax.imshow(percept_img, cmap="gray")
        else:
            ax.text(
                0.5,
                0.5,
                f"{implant}\npercepto no encontrado",
                ha="center",
                va="center",
            )

        ax.set_title(implant)
        ax.axis("off")

    fig.suptitle(
        f"sample_id={sample_id} | image_id={image_id} | ann_id={ann_id} | true={true_label}",
        fontsize=13,
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()

    return fig


def visualize_sample_predictions(
    df,
    sample_id,
    classes,
    implants,
    coco_images_dir,
    knn_results=None,
    cnn_models=None,
    cnn_gru_model=None,
    transform=None,
    device="cuda",
    knn_img_size=64,
    save_figure_path=None,
    sort_by_confidence=False,
):
    """
    Visualiza una muestra y muestra predicciones de:

    - KNN por implante
    - KNN multimplante
    - CNN por implante
    - CNN + GRU

    Parámetros esperados del notebook:
    - df: normalmente df_test
    - sample_id: ID de muestra a inspeccionar
    - classes: CLASSES
    - implants: IMPLANTS
    - coco_images_dir: carpeta val2017 de COCO
    - knn_results: diccionario de modelos KNN entrenados
    - cnn_models: models_a
    - cnn_gru_model: model_b
    - transform: val_tf
    - device: DEVICE
    """
    sample_rows = df[df["sample_id"] == sample_id].copy()

    if sample_rows.empty:
        raise ValueError(f"No existe sample_id={sample_id} en el dataframe.")

    true_label = sample_rows.iloc[0]["label"]

    _plot_images(
        sample_rows=sample_rows,
        sample_id=sample_id,
        classes=classes,
        implants=implants,
        coco_images_dir=coco_images_dir,
        save_path=save_figure_path,
    )

    predictions = _collect_predictions(
        sample_rows=sample_rows,
        classes=classes,
        implants=implants,
        knn_results=knn_results,
        cnn_models=cnn_models,
        cnn_gru_model=cnn_gru_model,
        transform=transform,
        device=device,
        knn_img_size=knn_img_size,
    )

    pred_df = _predictions_to_dataframe(
        predictions=predictions,
        classes=classes,
        true_label=true_label,
    )

    if sort_by_confidence and "confidence" in pred_df.columns:
        pred_df = pred_df.sort_values(
            by="confidence",
            ascending=False,
        ).reset_index(drop=True)

    display(pred_df)

    return pred_df


def visualize_random_sample_predictions(
    df,
    classes,
    implants,
    coco_images_dir,
    knn_results=None,
    cnn_models=None,
    cnn_gru_model=None,
    transform=None,
    device="cuda",
    knn_img_size=64,
    label=None,
    random_state=42,
    save_figure_path=None,
    sort_by_confidence=False,
):
    """
    Elige un sample_id aleatorio, opcionalmente filtrado por clase,
    y llama a visualize_sample_predictions.
    """
    df_aux = df.copy()

    if label is not None:
        df_aux = df_aux[df_aux["label"] == label].copy()

    if df_aux.empty:
        raise ValueError("No hay muestras disponibles con ese filtro.")

    sample_id = (
        df_aux["sample_id"]
        .drop_duplicates()
        .sample(1, random_state=random_state)
        .iloc[0]
    )

    print("sample_id seleccionado:", sample_id)

    return visualize_sample_predictions(
        df=df,
        sample_id=sample_id,
        classes=classes,
        implants=implants,
        coco_images_dir=coco_images_dir,
        knn_results=knn_results,
        cnn_models=cnn_models,
        cnn_gru_model=cnn_gru_model,
        transform=transform,
        device=device,
        knn_img_size=knn_img_size,
        save_figure_path=save_figure_path,
        sort_by_confidence=sort_by_confidence,
    )