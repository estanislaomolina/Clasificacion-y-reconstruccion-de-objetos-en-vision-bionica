import copy

import numpy as np
import torch
import torch.nn as nn


def compute_accuracy(logits, y):
    preds = logits.argmax(dim=1)
    correct = (preds == y).sum().item()
    total = y.numel()
    return correct / total


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        batch_size = y.size(0)

        running_loss += loss.item() * batch_size
        running_correct += (logits.argmax(dim=1) == y).sum().item()
        running_total += batch_size

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total

    return epoch_loss, epoch_acc


@torch.no_grad()
def eval_epoch(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        batch_size = y.size(0)

        running_loss += loss.item() * batch_size
        running_correct += (logits.argmax(dim=1) == y).sum().item()
        running_total += batch_size

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total

    return epoch_loss, epoch_acc


def fit(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    class_weights=None,
    label="model",
    monitor="val_acc",
    patience=5,
    min_delta=0.0,
):
    """
    Entrena un modelo con early stopping.

    monitor:
    - "val_acc": mejora cuando sube accuracy de validación
    - "val_loss": mejora cuando baja loss de validación

    patience:
    - cantidad de épocas consecutivas sin mejora antes de cortar
    """
    import copy
    import torch
    import torch.nn as nn

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    if monitor == "val_acc":
        best_score = -float("inf")
        mode = "max"
    elif monitor == "val_loss":
        best_score = float("inf")
        mode = "min"
    else:
        raise ValueError("monitor debe ser 'val_acc' o 'val_loss'.")

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_loss, val_acc = eval_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"[{label}] "
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        current_score = val_acc if monitor == "val_acc" else val_loss

        if mode == "max":
            improved = current_score > best_score + min_delta
        else:
            improved = current_score < best_score - min_delta

        if improved:
            best_score = current_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"[{label}] Early stopping en epoch {epoch}. "
                f"Mejor epoch: {best_epoch} | "
                f"best_{monitor}={best_score:.4f}"
            )
            break

    model.load_state_dict(best_state)

    history["best_epoch"] = best_epoch
    history["best_monitor"] = monitor
    history["best_score"] = float(best_score)
    history["early_stopping_patience"] = patience
    history["early_stopped"] = epoch < epochs
    history["epochs_ran"] = epoch

    return history


@torch.no_grad()
def predict_all(
    model,
    loader,
    device,
):
    """
    Devuelve y_true, y_pred como arrays NumPy.
    """
    model.eval()

    all_true = []
    all_pred = []

    for x, y in loader:
        x = x.to(device)

        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()

        all_pred.extend(preds.tolist())
        all_true.extend(y.numpy().tolist())

    return np.asarray(all_true), np.asarray(all_pred)


def summarize_lr_history(
    hist,
    lr,
    collapse_acc_threshold=0.20,
    loss_spike_threshold=2.0,
):
    """
    Resume el comportamiento de un entrenamiento para un learning rate.

    Métricas:
    - best_val_acc: mejor accuracy de validación
    - best_val_loss: menor loss de validación
    - collapse_count: cuántas épocas cayeron a accuracy muy baja
    - loss_spike_count: cuántas épocas tuvieron val_loss muy alto
    - val_acc_std_tail: variabilidad de val_acc en el último tercio
    - val_loss_std_tail: variabilidad de val_loss en el último tercio
    """
    val_acc = np.asarray(hist["val_acc"], dtype=float)
    val_loss = np.asarray(hist["val_loss"], dtype=float)

    n_epochs = len(val_acc)
    tail_size = max(3, n_epochs // 3)

    best_val_acc = float(val_acc.max())
    best_val_loss = float(val_loss.min())

    best_acc_epoch = int(val_acc.argmax()) + 1
    best_loss_epoch = int(val_loss.argmin()) + 1

    collapse_count = int((val_acc <= collapse_acc_threshold).sum())
    loss_spike_count = int((val_loss >= loss_spike_threshold).sum())

    val_acc_std_tail = float(val_acc[-tail_size:].std())
    val_loss_std_tail = float(val_loss[-tail_size:].std())

    return {
        "lr": float(lr),
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "best_acc_epoch": best_acc_epoch,
        "best_loss_epoch": best_loss_epoch,
        "collapse_count": collapse_count,
        "loss_spike_count": loss_spike_count,
        "val_acc_std_tail": val_acc_std_tail,
        "val_loss_std_tail": val_loss_std_tail,
    }


def select_lr_candidate(
    candidates,
    acc_tolerance=0.01,
    collapse_acc_threshold=0.20,
    loss_spike_threshold=2.0,
    prefer_lower_lr=True,
):
    """
    Selecciona el mejor candidato de LR.

    candidates debe ser una lista de diccionarios con:
    {
        "lr": lr,
        "model": model,
        "history": hist,
    }

    Criterio:
    1. Encuentra el mejor best_val_acc.
    2. Considera empatados los LR dentro de acc_tolerance.
    3. Entre los empatados, elige el más estable:
       - menos colapsos de val_acc
       - menos spikes de val_loss
       - menor variabilidad de val_loss al final
       - menor variabilidad de val_acc al final
       - menor best_val_loss
       - opcionalmente, menor LR
    """
    if not candidates:
        raise ValueError("La lista de candidatos está vacía.")

    enriched = []

    for cand in candidates:
        summary = summarize_lr_history(
            hist=cand["history"],
            lr=cand["lr"],
            collapse_acc_threshold=collapse_acc_threshold,
            loss_spike_threshold=loss_spike_threshold,
        )

        enriched.append({
            **cand,
            "summary": summary,
        })

    best_acc = max(c["summary"]["best_val_acc"] for c in enriched)

    eligible = [
        c for c in enriched
        if c["summary"]["best_val_acc"] >= best_acc - acc_tolerance
    ]

    def sorting_key(c):
        s = c["summary"]

        lr_key = s["lr"] if prefer_lower_lr else 0.0

        return (
            s["collapse_count"],
            s["loss_spike_count"],
            s["val_loss_std_tail"],
            s["val_acc_std_tail"],
            s["best_val_loss"],
            lr_key,
        )

    selected = sorted(eligible, key=sorting_key)[0]

    summaries = [c["summary"] for c in enriched]

    return selected, summaries