import torch.nn as nn
import torch


def dice_coefficient(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits).reshape(logits.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)
    inter = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    return ((2 * inter + eps) / (union + eps)).mean()


def iou_score(logits, targets, threshold=0.5, eps=1e-6):
    preds = (torch.sigmoid(logits) > threshold).float().reshape(logits.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)
    inter = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - inter
    return ((inter + eps) / (union + eps)).mean()


def pixel_accuracy(logits, targets, threshold=0.5):
    preds = (torch.sigmoid(logits) > threshold).float()
    return (preds == targets).float().mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = 1.0 - dice_coefficient(logits, targets)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss
    

def _ssim_map(x, y, window_size=11, eps=1e-8):
    """SSIM por canal, sin dependencias externas."""
    import torch.nn.functional as F_nn

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    kernel = torch.ones(1, 1, window_size, window_size, device=x.device) / (window_size ** 2)

    def _conv(t):
        # t: (B, 1, H, W)
        return F_nn.conv2d(t, kernel, padding=window_size // 2)

    mu_x = _conv(x)
    mu_y = _conv(y)
    sigma_x = _conv(x * x) - mu_x ** 2
    sigma_y = _conv(y * y) - mu_y ** 2
    sigma_xy = _conv(x * y) - mu_x * mu_y

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return num / (den + eps)


def ssim_score(pred, target, window_size=11):
    """
    pred, target: (B, 3, H, W) en [-1, 1].
    Devuelve SSIM promedio sobre batch y canales.
    """
    scores = []
    for c in range(pred.shape[1]):
        s = _ssim_map(
            pred[:, c:c+1, :, :],
            target[:, c:c+1, :, :],
            window_size=window_size,
        ).mean()
        scores.append(s)
    return torch.stack(scores).mean()


def psnr_score(pred, target, max_val=2.0):
    """
    pred, target en [-1, 1] → rango total = 2.0.
    Devuelve PSNR en dB.
    """
    mse = ((pred - target) ** 2).mean()
    if mse == 0:
        return torch.tensor(100.0)
    return 10 * torch.log10(max_val ** 2 / mse)


class L1SSIMLoss(nn.Module):
    """
    loss = l1_weight * L1 + (1 - l1_weight) * (1 - SSIM)
    """
    def __init__(self, l1_weight=0.8):
        super().__init__()
        self.l1_weight = l1_weight

    def forward(self, pred, target):
        l1 = torch.abs(pred - target).mean()
        ssim = ssim_score(pred, target)
        return self.l1_weight * l1 + (1 - self.l1_weight) * (1 - ssim)