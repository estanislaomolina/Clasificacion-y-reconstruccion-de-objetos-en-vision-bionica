from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from .losses import dice_coefficient, iou_score, pixel_accuracy


def plot_reconstruction_curves(history, title, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ["loss", "dice", "iou"]):
        ax.plot(history[f"train_{metric}"], label="train")
        ax.plot(history[f"val_{metric}"], label="val")
        ax.set_title(f"{title} - {metric}")
        ax.legend()
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120)
    plt.show()


@torch.no_grad()
def evaluate_reconstruction(model, loader, device, threshold=0.5):
    model.eval()
    dices, ious, accs = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        dices.append(dice_coefficient(logits, y).item())
        ious.append(iou_score(logits, y, threshold).item())
        accs.append(pixel_accuracy(logits, y, threshold).item())
    return {"dice": float(np.mean(dices)), "iou": float(np.mean(ious)), "pixel_acc": float(np.mean(accs))}


def plot_image_curves(history, title, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ["loss", "ssim", "psnr"]):
        ax.plot(history[f"train_{metric}"], label="train")
        ax.plot(history[f"val_{metric}"], label="val")
        ax.set_title(f"{title} - {metric}")
        ax.legend()
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120)
    plt.show()


@torch.no_grad()
def evaluate_image_reconstruction(model, loader, device):
    from .losses import ssim_score, psnr_score
    model.eval()
    ssims, psnrs, l1s = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        ssims.append(ssim_score(pred, y).item())
        psnrs.append(psnr_score(pred, y).item())
        l1s.append(torch.abs(pred - y).mean().item())
    return {
        "ssim": float(np.mean(ssims)),
        "psnr": float(np.mean(psnrs)),
        "l1": float(np.mean(l1s)),
    }