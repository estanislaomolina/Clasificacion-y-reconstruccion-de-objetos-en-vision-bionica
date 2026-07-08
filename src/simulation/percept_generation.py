from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

from pulse2percept.implants import ArgusII, PRIMA, AlphaAMS
from pulse2percept.models import ScoreboardModel
from pulse2percept.stimuli import ImageStimulus


IMPLANT_CLASSES = {
    "argus2": ArgusII,
    "prima": PRIMA,
    "alphams": AlphaAMS,
}


def letterbox_mask(mask_path, size=256):
    """
    Carga una máscara en escala de grises, mantiene el aspect ratio
    y agrega padding negro hasta obtener una imagen size x size.
    """
    mask_path = Path(mask_path)

    if not mask_path.exists():
        raise FileNotFoundError(f"No existe la máscara: {mask_path}")

    img = Image.open(mask_path).convert("L")
    img.thumbnail((size, size), Image.Resampling.NEAREST)

    canvas = Image.new("L", (size, size), 0)

    x = (size - img.width) // 2
    y = (size - img.height) // 2

    canvas.paste(img, (x, y))

    return canvas


def normalize_percept(frame):
    """
    Normaliza un percepto a uint8 en rango 0-255.
    """
    frame = np.asarray(frame)

    mn = frame.min()
    mx = frame.max()

    if mx > mn:
        frame = (frame - mn) / (mx - mn)
    else:
        frame = np.zeros_like(frame)

    return (frame * 255).astype(np.uint8)


def save_percept(frame, save_path):
    """
    Guarda un percepto normalizado como imagen PNG en escala de grises.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.fromarray(normalize_percept(frame), mode="L")
    img.save(save_path)


def stimulus_to_array(stim):
    """
    Convierte el estímulo redimensionado a la geometría del implante
    en un array NumPy.

    En este proyecto lo usamos como señal eléctrica / activación por electrodo.
    No representa todavía un tren temporal de pulsos bifásicos, sino la
    intensidad asignada a cada electrodo.
    """
    return np.asarray(stim.data)


def save_electrical_signal(
    signal,
    save_path,
    sample_id,
    image_id,
    ann_id,
    label,
    implant,
    mask_path,
    percept_path,
):
    """
    Guarda la señal eléctrica / activación por electrodo como .npz,
    junto con metadata mínima para trazabilidad.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        save_path,
        signal=signal,
        sample_id=str(sample_id),
        image_id=str(image_id),
        ann_id=str(ann_id),
        label=str(label),
        implant=str(implant),
        mask_path=str(mask_path),
        percept_path=str(percept_path),
    )


def build_scoreboard_model(
    percept_size=256,
    xrange=(-10, 10),
    yrange=(-10, 10),
):
    """
    Construye el modelo ScoreboardModel de pulse2percept.
    """
    xystep = (xrange[1] - xrange[0]) / (percept_size - 1)

    model = ScoreboardModel(
        xrange=xrange,
        yrange=yrange,
        xystep=xystep,
    )

    model.build()

    return model


def simulate_mask(mask_img, implant, model):
    """
    Simula un percepto a partir de una máscara y un implante.

    Devuelve:
    - percept_frame: imagen perceptual simulada.
    - signal_matrix: activación eléctrica por electrodo.
    """
    stim = ImageStimulus(np.array(mask_img))

    resized_stim = stim.resize(implant.shape)

    implant.stim = resized_stim

    signal_matrix = stimulus_to_array(resized_stim)

    percept = model.predict_percept(implant)

    percept_frame = percept.data[:, :, 0]

    return percept_frame, signal_matrix


def prepare_masks_dataframe(
    metadata_masks_file,
    target_classes,
    max_per_class=None,
    random_state=42,
):
    """
    Lee metadata_masks.csv, filtra las clases objetivo y opcionalmente
    limita la cantidad de muestras por clase.

    Espera un CSV con columnas:
    - sample_id
    - filename
    - class
    - image_id
    - ann_id
    """
    metadata_masks_file = Path(metadata_masks_file)

    if not metadata_masks_file.exists():
        raise FileNotFoundError(
            f"No existe metadata_masks.csv: {metadata_masks_file}"
        )

    df = pd.read_csv(metadata_masks_file)

    required_cols = {
        "sample_id",
        "filename",
        "class",
        "image_id",
        "ann_id",
    }

    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(
            f"Faltan columnas en metadata_masks.csv: {missing_cols}"
        )

    df = df[df["class"].isin(target_classes)].copy()

    if df.empty:
        raise ValueError(
            "No quedaron filas después de filtrar target_classes. "
            "Revisá los nombres de las clases."
        )

    df = df.rename(columns={"class": "label"})

    if max_per_class is not None:
        sampled_groups = []

        for label, group in df.groupby("label"):
            n = min(max_per_class, len(group))
            sampled = group.sample(n=n, random_state=random_state)
            sampled_groups.append(sampled)

        df = pd.concat(sampled_groups, axis=0).reset_index(drop=True)

    return df


def resolve_mask_path(row, masks_dir):
    """
    Devuelve el path correcto de la máscara.

    Primero intenta usar el path guardado en metadata_masks.csv.
    Si ese path no existe, reconstruye el path usando:

    masks_dir / label / sample_id.png
    """
    original_path = Path(row["filename"])

    if original_path.exists():
        return original_path

    class_dir = str(row["label"]).replace(" ", "_")

    reconstructed_path = (
        Path(masks_dir)
        / class_dir
        / f"{row['sample_id']}.png"
    )

    return reconstructed_path


def generate_percepts(
    metadata_masks_file,
    output_percepts_dir,
    output_signals_dir,
    output_metadata_file,
    masks_dir,
    target_classes,
    implants=("argus2", "prima", "alphams"),
    max_per_class=900,
    input_size=256,
    percept_size=256,
    xrange=(-10, 10),
    yrange=(-10, 10),
    random_state=42,
    skip_existing=True,
):
    """
    Genera perceptos simulados y señales eléctricas para máscaras COCO.

    Entradas:
    - metadata_masks_file:
        data/metadata/metadata_masks.csv

    - output_percepts_dir:
        data/percepts/

    - output_signals_dir:
        data/electrical_signals/

    - output_metadata_file:
        data/metadata/metadata_percepts.csv

    - masks_dir:
        data/masks/

    Salidas:
    - perceptos PNG:
        data/percepts/{implant}/{class}/{sample_id}.png

    - señales eléctricas NPZ:
        data/electrical_signals/{implant}/{class}/{sample_id}.npz

    - metadata:
        metadata_percepts.csv con columnas:
        sample_id, image_id, ann_id, label, implant,
        mask_path, signal_path, percept_path
    """
    output_percepts_dir = Path(output_percepts_dir)
    output_signals_dir = Path(output_signals_dir)
    output_metadata_file = Path(output_metadata_file)
    masks_dir = Path(masks_dir)

    output_percepts_dir.mkdir(parents=True, exist_ok=True)
    output_signals_dir.mkdir(parents=True, exist_ok=True)
    output_metadata_file.parent.mkdir(parents=True, exist_ok=True)

    invalid_implants = set(implants) - set(IMPLANT_CLASSES.keys())

    if invalid_implants:
        raise ValueError(f"Implantes no reconocidos: {invalid_implants}")

    df = prepare_masks_dataframe(
        metadata_masks_file=metadata_masks_file,
        target_classes=target_classes,
        max_per_class=max_per_class,
        random_state=random_state,
    )

    print("Máscaras seleccionadas:")
    print(df["label"].value_counts())
    print("Total:", len(df))

    print("\nConstruyendo ScoreboardModel...")
    model = build_scoreboard_model(
        percept_size=percept_size,
        xrange=xrange,
        yrange=yrange,
    )
    print("Modelo listo.")

    rows = []

    saved = 0
    reused = 0
    skipped = 0
    errors = 0

    total = len(df)

    for implant_name in implants:
        print("\n" + "=" * 60)
        print(f"Implante: {implant_name}")
        print("=" * 60)

        ImplantClass = IMPLANT_CLASSES[implant_name]
        implant = ImplantClass()

        for idx, row in df.iterrows():
            mask_path = resolve_mask_path(row, masks_dir)

            if not mask_path.exists():
                skipped += 1
                print(f"No existe máscara, se saltea: {mask_path}")
                continue

            class_dir = str(row["label"]).replace(" ", "_")

            percept_path = (
                output_percepts_dir
                / implant_name
                / class_dir
                / mask_path.name
            )

            signal_path = (
                output_signals_dir
                / implant_name
                / class_dir
                / f"{mask_path.stem}.npz"
            )

            if skip_existing and percept_path.exists() and signal_path.exists():
                reused += 1

                rows.append({
                    "sample_id": row["sample_id"],
                    "image_id": row["image_id"],
                    "ann_id": row["ann_id"],
                    "label": row["label"],
                    "implant": implant_name,
                    "mask_path": str(mask_path),
                    "signal_path": str(signal_path),
                    "percept_path": str(percept_path),
                })

                continue

            try:
                mask_img = letterbox_mask(mask_path, size=input_size)

                percept, signal_matrix = simulate_mask(
                    mask_img=mask_img,
                    implant=implant,
                    model=model,
                )

                if saved == 0:
                    print("Tamaño del percepto:", percept.shape)
                    print("Tamaño de la señal:", signal_matrix.shape)

                save_percept(percept, percept_path)

                save_electrical_signal(
                    signal=signal_matrix,
                    save_path=signal_path,
                    sample_id=row["sample_id"],
                    image_id=row["image_id"],
                    ann_id=row["ann_id"],
                    label=row["label"],
                    implant=implant_name,
                    mask_path=mask_path,
                    percept_path=percept_path,
                )

                rows.append({
                    "sample_id": row["sample_id"],
                    "image_id": row["image_id"],
                    "ann_id": row["ann_id"],
                    "label": row["label"],
                    "implant": implant_name,
                    "mask_path": str(mask_path),
                    "signal_path": str(signal_path),
                    "percept_path": str(percept_path),
                })

                saved += 1

            except Exception as e:
                errors += 1
                print(f"Error con {mask_path.name} - {implant_name}: {e}")

            current = idx + 1

            if current % 100 == 0:
                print(f"{current}/{total}")

    if not rows:
        raise RuntimeError(
            "No se generó ningún percepto. Revisá paths, metadata o pulse2percept."
        )

    df_out = pd.DataFrame(rows)

    df_out = df_out.sort_values(
        by=["sample_id", "implant"]
    ).reset_index(drop=True)

    df_out.to_csv(output_metadata_file, index=False)

    counts = (
        df_out.groupby(["implant", "label"])
              .size()
              .reset_index(name="n")
    )

    sample_implant_counts = (
        df_out.groupby("sample_id")["implant"]
              .nunique()
              .value_counts()
              .sort_index()
              .to_dict()
    )

    summary = {
        "selected_masks": len(df),
        "metadata_file": str(output_metadata_file),
        "output_percepts_dir": str(output_percepts_dir),
        "output_signals_dir": str(output_signals_dir),
        "saved": saved,
        "reused": reused,
        "skipped": skipped,
        "errors": errors,
        "rows": len(df_out),
        "counts": counts,
        "sample_implant_counts": sample_implant_counts,
    }

    return summary


def print_percept_summary(summary):
    """
    Imprime un resumen limpio de la generación de perceptos y señales.
    """
    print("\nListo.")
    print(f"Máscaras seleccionadas: {summary['selected_masks']}")
    print(f"Perceptos/señales nuevos guardados: {summary['saved']}")
    print(f"Perceptos/señales reutilizados: {summary['reused']}")
    print(f"Máscaras salteadas: {summary['skipped']}")
    print(f"Errores: {summary['errors']}")
    print(f"Filas metadata: {summary['rows']}")
    print(f"Metadata: {summary['metadata_file']}")
    print(f"Percepts dir: {summary['output_percepts_dir']}")
    print(f"Signals dir: {summary['output_signals_dir']}")

    print("\nDistribución por implante y clase:")
    print(summary["counts"])

    print("\nCantidad de implantes disponibles por sample_id:")
    print(summary["sample_implant_counts"])


def load_electrical_signal(signal_path):
    """
    Carga una señal eléctrica guardada como .npz.

    Devuelve:
    - signal: array NumPy
    - metadata: diccionario con sample_id, image_id, ann_id, label, implant, etc.
    """
    signal_path = Path(signal_path)

    if not signal_path.exists():
        raise FileNotFoundError(f"No existe la señal: {signal_path}")

    data = np.load(signal_path)

    signal = data["signal"]

    metadata = {
        "sample_id": str(data["sample_id"]),
        "image_id": str(data["image_id"]),
        "ann_id": str(data["ann_id"]),
        "label": str(data["label"]),
        "implant": str(data["implant"]),
        "mask_path": str(data["mask_path"]),
        "percept_path": str(data["percept_path"]),
    }

    return signal, metadata