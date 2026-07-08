import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """
    Encoder CNN simple para perceptos en escala de grises.
    Entrada:  [B, 1, H, W]
    Salida:   [B, feat_dim]
    """

    def __init__(self, feat_dim=128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.proj(x)
        return x


class CNNClassifier(nn.Module):
    """
    Clasificador CNN para un solo percepto.
    """

    def __init__(self, num_classes, feat_dim=128):
        super().__init__()

        self.encoder = CNNEncoder(feat_dim=feat_dim)

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        feat = self.encoder(x)
        logits = self.classifier(feat)
        return logits


class CNNGRUClassifier(nn.Module):
    """
    Modelo multimplante.

    Entrada:
    x: [B, T, 1, H, W]

    T corresponde al número de implantes:
    argus2, prima, alphams.
    """

    def __init__(
        self,
        num_classes,
        feat_dim=128,
        hidden_dim=128,
        num_layers=1,
        bidirectional=False,
    ):
        super().__init__()

        self.encoder = CNNEncoder(feat_dim=feat_dim)

        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        gru_out_dim = hidden_dim * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """
        x: [B, T, 1, H, W]
        """
        batch_size, seq_len, channels, height, width = x.shape

        x = x.view(batch_size * seq_len, channels, height, width)

        feats = self.encoder(x)

        feats = feats.view(batch_size, seq_len, -1)

        gru_out, h_n = self.gru(feats)

        last_out = gru_out[:, -1, :]

        logits = self.classifier(last_out)

        return logits