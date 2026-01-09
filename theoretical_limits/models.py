import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam


logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


"""
Simple linear model.
"""

class SimpleLinearModel(pl.LightningModule):
    
    def __init__(self, input_dim: int, num_classes: int, learning_rate: float = 1e-3):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.lr = learning_rate
        
    def forward(self, x):
        return self.linear(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)


"""
scVI style model.
"""

class MLP(nn.Module):
    """Simple fully-connected block: Linear -> BatchNorm -> ReLU -> Dropout"""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Encoder(nn.Module):
    """Encoder producing mu and logvar for the latent distribution."""

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(MLP(prev, h, dropout))
            prev = h
        self.feature_extractor = nn.Sequential(*layers) if layers else nn.Identity()
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        h = self.feature_extractor(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """Decoder mapping latent z back to input space (reconstruction)."""

    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            layers.append(MLP(prev, h, dropout))
            prev = h
        self.net = nn.Sequential(*layers) if layers else nn.Identity()
        self.final = nn.Linear(prev, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.net(z)
        recon = self.final(h)
        return recon


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL divergence between N(mu, var) and N(0, I) per sample.
    returns shape (batch,) -> sum across latent dims.
    KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


class SCVI(pl.LightningModule):
    """PyTorch Lightning module for an scVI-like VAE using MSE reconstruction loss.

    Args:
        input_dim: dimensionality of input features (e.g., number of genes)
        hidden_dims: list of hidden layer sizes for encoder/decoder
        latent_dim: size of latent representation z
        lr: learning rate
        weight_decay: optimizer weight decay
        beta: scaling applied to KL term (beta-VAE style). Default 1.0
        dropout: dropout probability for MLP blocks
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        latent_dim: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        beta: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.encoder = Encoder(input_dim, hidden_dims, latent_dim, dropout)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim, dropout)

    def encode(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict:
        """Returns a dict with keys: recon (reconstruction), mu, logvar, z"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return {"recon": recon, "mu": mu, "logvar": logvar, "z": z}

    def step(self, batch: torch.Tensor) -> dict:
        """Single step computing recon, mse, kl and total loss."""
        x = batch if isinstance(batch, torch.Tensor) else batch[0]
        out = self.forward(x)
        recon = out["recon"]
        mu = out["mu"]
        logvar = out["logvar"]
        mse_elementwise = F.mse_loss(recon, x, reduction="none")
        mse_per_sample = mse_elementwise.mean(dim=1)
        mse = mse_per_sample.mean()
        kl_per_sample = kl_divergence(mu, logvar)
        kl = kl_per_sample.mean()
        loss = mse + self.hparams.beta * kl

        return loss 

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return opt
