from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_percept_vector(percept_path, img_size=64):
    """
    Carga un percepto, lo redimensiona y lo aplana.
    """
    percept_path = Path(percept_path)

    if not percept_path.exists():
        raise FileNotFoundError(f"No existe percept_path: {percept_path}")

    img = Image.open(percept_path).convert("L")
    img = img.resize((img_size, img_size), Image.Resampling.BILINEAR)

    arr = np.asarray(img, dtype=np.float32) / 255.0

    return arr.reshape(-1)


def build_single_implant_matrix(
    df,
    implant,
    classes,
    img_size=64,
):
    """
    Construye X, y para un único implante.
    """
    df_imp = df[df["implant"] == implant].copy()

    if df_imp.empty:
        raise ValueError(f"No hay filas para implant={implant}")

    cls2idx = {cls: i for i, cls in enumerate(classes)}

    X = []
    y = []
    sample_ids = []

    for _, row in df_imp.iterrows():
        X.append(
            load_percept_vector(
                row["percept_path"],
                img_size=img_size,
            )
        )
        y.append(cls2idx[row["label"]])
        sample_ids.append(row["sample_id"])

    X = np.stack(X)
    y = np.asarray(y)

    return X, y, sample_ids


def build_multi_implant_matrix(
    df,
    implants,
    classes,
    img_size=64,
):
    """
    Construye X, y concatenando perceptos de varios implantes
    para cada sample_id.

    Requiere que cada sample_id tenga todos los implantes.
    """
    cls2idx = {cls: i for i, cls in enumerate(classes)}

    pivot = (
        df.pivot_table(
            index="sample_id",
            columns="implant",
            values="percept_path",
            aggfunc="first",
        )
        .dropna(subset=implants)
        .reset_index()
    )

    if pivot.empty:
        raise ValueError(
            "No hay sample_id con todos los implantes. "
            "Revisá metadata_percepts.csv."
        )

    label_map = (
        df.drop_duplicates("sample_id")
        .set_index("sample_id")["label"]
    )

    pivot["label"] = pivot["sample_id"].map(label_map)

    X = []
    y = []
    sample_ids = []

    for _, row in pivot.iterrows():
        features = []

        for implant in implants:
            vec = load_percept_vector(
                row[implant],
                img_size=img_size,
            )
            features.append(vec)

        X.append(np.concatenate(features))
        y.append(cls2idx[row["label"]])
        sample_ids.append(row["sample_id"])

    X = np.stack(X)
    y = np.asarray(y)

    return X, y, sample_ids


def make_knn_pipeline(
    n_components=50,
    n_neighbors=5,
    weights="distance",
    metric="euclidean",
):
    """
    Pipeline:
    StandardScaler -> PCA -> KNN
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=42)),
        ("knn", KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
        )),
    ])


def tune_knn_on_validation(
    X_train,
    y_train,
    X_val,
    y_val,
    n_components_candidates=(25, 50, 100),
    k_candidates=(1, 3, 5, 7, 11),
    weights_candidates=("uniform", "distance"),
    metric_candidates=("euclidean", "manhattan"),
):
    """
    Busca hiperparámetros usando validación.
    No toca test.
    """
    max_components = min(X_train.shape[0], X_train.shape[1])

    results = []
    best_model = None
    best_score = -1.0
    best_params = None

    for n_components in n_components_candidates:
        if n_components > max_components:
            continue

        for k in k_candidates:
            if k > len(y_train):
                continue

            for weights in weights_candidates:
                for metric in metric_candidates:
                    model = make_knn_pipeline(
                        n_components=n_components,
                        n_neighbors=k,
                        weights=weights,
                        metric=metric,
                    )

                    model.fit(X_train, y_train)

                    val_pred = model.predict(X_val)
                    val_acc = accuracy_score(y_val, val_pred)

                    params = {
                        "n_components": n_components,
                        "n_neighbors": k,
                        "weights": weights,
                        "metric": metric,
                        "val_accuracy": val_acc,
                    }

                    results.append(params)

                    if val_acc > best_score:
                        best_score = val_acc
                        best_model = model
                        best_params = params

    results_df = pd.DataFrame(results)

    if best_model is None:
        raise RuntimeError(
            "No se pudo entrenar ningún KNN. "
            "Revisá n_components_candidates y k_candidates."
        )

    return best_model, best_params, results_df


def evaluate_knn_model(
    model,
    X_test,
    y_test,
    classes,
):
    """
    Evalúa un modelo KNN en test.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_test,
        y_pred,
        target_names=classes,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": acc,
        "y_pred": y_pred,
        "report_dict": report_dict,
        "report_text": report_text,
        "confusion_matrix": cm,
    }


def run_knn_single_implant(
    df_train,
    df_val,
    df_test,
    implant,
    classes,
    img_size=64,
    n_components_candidates=(25, 50, 100),
    k_candidates=(1, 3, 5, 7, 11),
):
    """
    Entrena y evalúa KNN para un solo implante.
    """
    X_train, y_train, _ = build_single_implant_matrix(
        df_train,
        implant=implant,
        classes=classes,
        img_size=img_size,
    )

    X_val, y_val, _ = build_single_implant_matrix(
        df_val,
        implant=implant,
        classes=classes,
        img_size=img_size,
    )

    X_test, y_test, _ = build_single_implant_matrix(
        df_test,
        implant=implant,
        classes=classes,
        img_size=img_size,
    )

    model, best_params, val_results = tune_knn_on_validation(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_components_candidates=n_components_candidates,
        k_candidates=k_candidates,
    )

    test_results = evaluate_knn_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        classes=classes,
    )

    return {
        "model": model,
        "best_params": best_params,
        "val_results": val_results,
        "test_results": test_results,
    }


def run_knn_multi_implant(
    df_train,
    df_val,
    df_test,
    implants,
    classes,
    img_size=64,
    n_components_candidates=(25, 50, 100),
    k_candidates=(1, 3, 5, 7, 11),
):
    """
    Entrena y evalúa KNN concatenando perceptos de varios implantes.
    """
    X_train, y_train, _ = build_multi_implant_matrix(
        df_train,
        implants=implants,
        classes=classes,
        img_size=img_size,
    )

    X_val, y_val, _ = build_multi_implant_matrix(
        df_val,
        implants=implants,
        classes=classes,
        img_size=img_size,
    )

    X_test, y_test, _ = build_multi_implant_matrix(
        df_test,
        implants=implants,
        classes=classes,
        img_size=img_size,
    )

    model, best_params, val_results = tune_knn_on_validation(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_components_candidates=n_components_candidates,
        k_candidates=k_candidates,
    )

    test_results = evaluate_knn_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        classes=classes,
    )

    return {
        "model": model,
        "best_params": best_params,
        "val_results": val_results,
        "test_results": test_results,
    }