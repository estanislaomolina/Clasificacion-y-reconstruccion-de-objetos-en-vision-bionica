from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch


@torch.no_grad()
def visualize_reconstructions(model, dataset, device, n_samples=6, threshold=0.5, save_path=None, title=""):
    model.eval()
    idxs = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    fig, axes = plt.subplots(len(idxs), 3, figsize=(9, 3 * len(idxs)))
    for row, idx in enumerate(idxs):
        x, y = dataset[idx]
        logits = model(x.unsqueeze(0).to(device))
        pred = (torch.sigmoid(logits) > threshold).float().cpu().squeeze().numpy()

        percept_img = x[0].cpu().numpy()  # primer canal (alcanza como referencia visual)
        mask_img = y.squeeze().cpu().numpy()

        ax_row = axes[row] if len(idxs) > 1 else axes
        for ax, img, t in zip(ax_row, [percept_img, pred, mask_img], ["Percepto", "Reconstrucción", "Máscara original"]):
            ax.imshow(img, cmap="gray")
            ax.set_title(t)
            ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120)
    plt.show()


@torch.no_grad()
def visualize_image_reconstructions(model, dataset, device, n_samples=4, save_path=None, title=""):
    """
    Muestra: percepto (gris) | reconstrucción RGB | imagen original RGB.
    """
    model.eval()
    idxs = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    fig, axes = plt.subplots(len(idxs), 3, figsize=(10, 3.5 * len(idxs)))
    if len(idxs) == 1:
        axes = [axes]

    for row_i, idx in enumerate(idxs):
        x, y = dataset[idx]
        pred = model(x.unsqueeze(0).to(device)).cpu().squeeze(0)

        # desnormalizar de [-1,1] a [0,1]
        percept_np = (x[0].numpy() * 0.5 + 0.5).clip(0, 1)
        pred_np = (pred.permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)
        target_np = (y.permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)

        for ax, img, t, cmap in zip(
            axes[row_i],
            [percept_np, pred_np, target_np],
            ["Percepto", "Reconstrucción", "Original COCO"],
            ["gray", None, None],
        ):
            ax.imshow(img, cmap=cmap)
            ax.set_title(t)
            ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120)
    plt.show()