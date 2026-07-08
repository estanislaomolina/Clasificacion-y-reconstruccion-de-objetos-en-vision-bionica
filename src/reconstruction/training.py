import copy
import time

import torch

from .losses import BCEDiceLoss, dice_coefficient, iou_score, pixel_accuracy, L1SSIMLoss, ssim_score, psnr_score


def _run_epoch(model, loader, device, optimizer, loss_fn):
    train = optimizer is not None
    model.train() if train else model.eval()

    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "pixel_acc": 0.0}
    n_batches = 0

    with torch.enable_grad() if train else torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()

            logits = model(x)
            loss = loss_fn(logits, y)

            if train:
                loss.backward()
                optimizer.step()

            totals["loss"] += loss.item()
            totals["dice"] += dice_coefficient(logits, y).item()
            totals["iou"] += iou_score(logits, y).item()
            totals["pixel_acc"] += pixel_accuracy(logits, y).item()
            n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


def fit_reconstruction(
    model, train_loader, val_loader, device, epochs, lr,
    label="model", monitor="val_dice", patience=5, min_delta=0.0, bce_weight=0.5,
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = BCEDiceLoss(bce_weight=bce_weight)

    history = {f"{split}_{metric}": [] for split in ["train", "val"] for metric in ["loss", "dice", "iou", "pixel_acc"]}

    best_metric, best_state, no_improve = -float("inf"), None, 0
    monitor_key = monitor.replace("val_", "")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_stats = _run_epoch(model, train_loader, device, optimizer, loss_fn)
        val_stats = _run_epoch(model, val_loader, device, None, loss_fn)

        for k, v in train_stats.items():
            history[f"train_{k}"].append(v)
        for k, v in val_stats.items():
            history[f"val_{k}"].append(v)

        current = val_stats[monitor_key]
        print(
            f"[{label}] epoch {epoch:02d}/{epochs} "
            f"train_loss={train_stats['loss']:.4f} val_loss={val_stats['loss']:.4f} "
            f"val_dice={val_stats['dice']:.4f} val_iou={val_stats['iou']:.4f} "
            f"({time.time() - t0:.1f}s)"
        )

        if current > best_metric + min_delta:
            best_metric, best_state, no_improve = current, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"[{label}] early stopping en epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history[f"best_{monitor}"] = best_metric
    return model, history


def select_lr_candidate_recon(candidates, monitor="val_dice", collapse_threshold=0.05, tolerance=0.01, prefer_lower_lr=True):
    key = monitor
    scored = [{**c, "best_val_metric": max(c["history"][key])} for c in candidates]

    valid = [c for c in scored if c["best_val_metric"] >= collapse_threshold]
    pool = valid if valid else scored

    best_overall = max(c["best_val_metric"] for c in pool)
    close_enough = [c for c in pool if best_overall - c["best_val_metric"] <= tolerance]

    selected = min(close_enough, key=lambda c: c["lr"]) if prefer_lower_lr else max(close_enough, key=lambda c: c["best_val_metric"])
    summaries = [{"lr": c["lr"], f"best_{monitor}": c["best_val_metric"]} for c in scored]

    return selected, summaries


def _run_image_epoch(model, loader, device, optimizer, loss_fn):
    train = optimizer is not None
    model.train() if train else model.eval()

    totals = {"loss": 0.0, "ssim": 0.0, "psnr": 0.0, "l1": 0.0}
    n_batches = 0

    with torch.enable_grad() if train else torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()

            pred = model(x)
            loss = loss_fn(pred, y)

            if train:
                loss.backward()
                optimizer.step()

            totals["loss"] += loss.item()
            totals["ssim"] += ssim_score(pred, y).item()
            totals["psnr"] += psnr_score(pred, y).item()
            totals["l1"] += torch.abs(pred - y).mean().item()
            n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


def fit_image_reconstruction(
    model, train_loader, val_loader, device, epochs, lr,
    label="model", monitor="val_ssim", patience=5,
    min_delta=0.0, l1_weight=0.8,
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = L1SSIMLoss(l1_weight=l1_weight)

    history = {
        f"{split}_{metric}": []
        for split in ["train", "val"]
        for metric in ["loss", "ssim", "psnr", "l1"]
    }

    best_metric, best_state, no_improve = -float("inf"), None, 0
    monitor_key = monitor.replace("val_", "")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_stats = _run_image_epoch(model, train_loader, device, optimizer, loss_fn)
        val_stats = _run_image_epoch(model, val_loader, device, None, loss_fn)

        for k, v in train_stats.items():
            history[f"train_{k}"].append(v)
        for k, v in val_stats.items():
            history[f"val_{k}"].append(v)

        current = val_stats[monitor_key]
        print(
            f"[{label}] epoch {epoch:02d}/{epochs} "
            f"train_loss={train_stats['loss']:.4f} val_loss={val_stats['loss']:.4f} "
            f"val_ssim={val_stats['ssim']:.4f} val_psnr={val_stats['psnr']:.2f}dB "
            f"({time.time() - t0:.1f}s)"
        )

        if current > best_metric + min_delta:
            best_metric = current
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"[{label}] early stopping en epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history[f"best_{monitor}"] = best_metric
    return model, history


def select_lr_candidate_img(candidates, monitor="val_ssim", collapse_threshold=0.3, tolerance=0.01, prefer_lower_lr=True):
    key = monitor.replace("val_", "")
    scored = [{**c, "best_val_metric": max(c["history"][f"val_{key}"])} for c in candidates]

    valid = [c for c in scored if c["best_val_metric"] >= collapse_threshold]
    pool = valid if valid else scored

    best_overall = max(c["best_val_metric"] for c in pool)
    close_enough = [c for c in pool if best_overall - c["best_val_metric"] <= tolerance]

    selected = min(close_enough, key=lambda c: c["lr"]) if prefer_lower_lr else max(close_enough, key=lambda c: c["best_val_metric"])
    summaries = [{"lr": c["lr"], f"best_{monitor}": c["best_val_metric"]} for c in scored]

    return selected, summaries