#!/usr/bin/env python3
# ================================================================
# PHYSICAL AUTOENCODER - RED 1
#
# Cambios incluidos:
#   - Geometría oficial: 260 x 1560 x 256
#   - Cubo físico: 260 x 1560 x 258
#   - Parches: 32 x 32
#   - Separación espacial: 70 / 15 / 15
#   - Padding reflectante
#   - Entrenamiento con Lightning
#   - Barrido automático de dimensiones del vector latente
#   - 3 pérdidas: emisividad, temperatura y profundidad
#   - Early stopping + mejor checkpoint
#   - Métricas en unidades físicas
#   - Validación visual sobre el mismo parche para todos los latentes
#
#
# Archivos esperados:
#   data/processed/emissivity.npy  -> (260,1560,256)
#   data/processed/temperature.npy -> (260,1560)
#   data/processed/depth.npy       -> (260,1560)
# ================================================================

import argparse
import gc
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

import shutil
from scipy.ndimage import gaussian_filter
import subprocess
import sys


H_EXPECTED = 260
W_EXPECTED = 1560
N_EMISSIVITY = 256
N_PHYSICAL = 258

DRIVE_FOLDER_ID = "1ajvvULL8PoqWUeLtzkqPDkw1ZXnJyi4F"
REQUIRED_DATA_FILES = [
    "emissivity_map.npy",
    "fixed_temperature.npy",
    "depth_completion.npy",
]

def ensure_extracted_data(
    data_dir,
    folder_id=DRIVE_FOLDER_ID,
    required=REQUIRED_DATA_FILES,
):
    """
    Descarga la carpeta pública de Drive dentro de `data_dir`
    (se crea si no existe) y devuelve la carpeta que contiene los
    archivos requeridos. Si ya están, no descarga nada.
    """
    data_dir = Path(data_dir)

    def locate():
        if all((data_dir / f).exists() for f in required):
            return data_dir
        if data_dir.exists():
            for sub in data_dir.rglob("*"):
                if sub.is_dir() and all((sub / f).exists() for f in required):
                    return sub
        return None

    found = locate()
    if found is not None:
        return found

    try:
        import gdown
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "gdown"]
        )
        import gdown

    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDescargando datos desde Drive -> {data_dir}")

    gdown.download_folder(
        id=folder_id,
        output=str(data_dir),
        quiet=False,
        use_cookies=False,
    )

    found = locate()
    if found is None:
        raise FileNotFoundError(
            f"Tras la descarga no se encontraron {required} en {data_dir}. "
            "Verifica que la carpeta de Drive sea pública."
        )

    return found

# Colormaps por modalidad para las visualizaciones (save_visual_validation
# y save_full_scene_reconstruction). Se aplican a los paneles "Original" y
# "Reconstruida"; los paneles de error quedan con el colormap por defecto.
CMAP_EMISSIVITY = "viridis_r"
CMAP_TEMPERATURE = "inferno"
CMAP_DEPTH = "viridis"


# ================================================================
# ARGUMENTOS
# ================================================================

def parse_args():
    p = argparse.ArgumentParser()

    # Barrido: de mayor a menor.
    p.add_argument(
        "--latent-dims",
        nargs="+",
        type=int,
        default=[2048],
    )

    p.add_argument("--patch-size", type=int, default=32)

    p.add_argument("--data-dir", default="extracted_data")
    p.add_argument("--drive-folder-id", default=DRIVE_FOLDER_ID)
    p.add_argument("--output-dir", default="outputs/physical_autoencoder")

    # Con stride 8 se obtienen muchos más parches de validación/test.
    p.add_argument("--train-stride", type=int, default=8)
    p.add_argument("--val-stride", type=int, default=8)
    p.add_argument("--test-stride", type=int, default=8)

    p.add_argument("--train-fraction", type=float, default=0.70)
    p.add_argument("--val-fraction", type=float, default=0.15)

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)

    # Métrica para early stopping y mejor checkpoint.
    p.add_argument(
        "--monitor",
        default="val_mae",
        choices=["val_mae", "val_mse"],
    )

    # Skip connections tipo U-Net
    p.add_argument("--skip-connections", type=bool, default=True)

    # Pesos de las tres pérdidas.
    p.add_argument("--weight-eps", type=float, default=1.0)
    p.add_argument("--weight-temp", type=float, default=10.0)
    p.add_argument("--weight-depth", type=float, default=10.0)
    p.add_argument("--edge-alpha", type=float, default=0.25)

    # 5% por defecto frente al modelo de 2048.
    p.add_argument("--tolerance", type=float, default=0)

    # Siempre se compara visualmente el mismo parche.
    p.add_argument("--visual-patch-index", type=int, default=0)

    # Reconstrucción de la escena completa.
    p.add_argument("--scene-stride", type=int, default=16)
    p.add_argument("--scene-bands", nargs="+", type=int, default=[0])

    p.add_argument("--seed", type=int, default=2026)

    return p.parse_args()


# ================================================================
# DATOS
# ================================================================

def load_data(data_dir: Path):
    eps_path = data_dir / "emissivity_map.npy"
    temp_path = data_dir / "fixed_temperature.npy"
    depth_path = data_dir / "depth_completion.npy"

    for path in [eps_path, temp_path, depth_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró {path}. "
                "Exporta primero los resultados del notebook."
            )

    emissivity = np.load(eps_path).astype(np.float32, copy=False)
    temperature = np.load(temp_path).astype(np.float32, copy=False)
    depth = np.load(depth_path).astype(np.float32, copy=False)

    if temperature.ndim == 3 and temperature.shape[-1] == 1:
        temperature = temperature[..., 0]

    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    if emissivity.shape != (H_EXPECTED, W_EXPECTED, N_EMISSIVITY):
        raise ValueError(
            f"Emisividad incorrecta. "
            f"Esperado {(H_EXPECTED,W_EXPECTED,N_EMISSIVITY)}, "
            f"recibido {emissivity.shape}"
        )

    if temperature.shape != (H_EXPECTED, W_EXPECTED):
        raise ValueError(
            f"Temperatura incorrecta. "
            f"Esperado {(H_EXPECTED,W_EXPECTED)}, "
            f"recibido {temperature.shape}"
        )

    if depth.shape != (H_EXPECTED, W_EXPECTED):
        raise ValueError(
            f"Profundidad incorrecta. "
            f"Esperado {(H_EXPECTED,W_EXPECTED)}, "
            f"recibido {depth.shape}"
        )

    for name, arr in [
        ("emissivity", emissivity),
        ("temperature", temperature),
        ("depth", depth),
    ]:
        if not np.isfinite(arr).all():
            raise ValueError(f"{name}.npy contiene NaN o Inf.")

    physical_cube = physical_cube = np.concatenate(
                                    [
                                        emissivity,
                                        temperature[..., None],
                                        depth[..., None],
                                    ],
                                        axis=2,
                                    ).astype(np.float32, copy=False)

    assert physical_cube.shape == (
        H_EXPECTED,
        W_EXPECTED,
        N_PHYSICAL,
    )

    print("\n========== DATOS ==========")
    print("Emisividad :", emissivity.shape)
    print("Temperatura:", temperature.shape)
    print("Profundidad:", depth.shape)
    print("Cubo físico:", physical_cube.shape)

    return physical_cube


def spatial_regions(width, train_fraction, val_fraction):
    val_end = int(width * val_fraction)
    test_end = width - int(width * train_fraction)

    if not (0 < val_end < test_end < width):
        raise ValueError("División train/val/test inválida.")

    return {
        "train": (test_end, width),
        "val": (0, val_end),
        "test": (val_end, test_end),
    }


def compute_normalization(cube, regions):
    x0, x1 = regions["train"]
    train = cube[:, x0:x1, :]

    eps = train[:, :, :256]
    temp = train[:, :, 256]
    depth = train[:, :, 257]

    eps_mean = np.mean(eps, axis=(0, 1), keepdims=True)
    eps_std = np.std(eps, axis=(0, 1), keepdims=True) + 1e-6

    temp_mean = float(np.mean(temp))
    temp_std = float(np.std(temp) + 1e-6)

    depth_mean = float(np.mean(depth))
    depth_std = float(np.std(depth) + 1e-6)

    return {
        "eps_mean": eps_mean.astype(np.float32),
        "eps_std": eps_std.astype(np.float32),
        "temp_mean": temp_mean,
        "temp_std": temp_std,
        "depth_mean": depth_mean,
        "depth_std": depth_std,
    }


def normalize_cube(cube, stats):
    out = np.empty_like(cube, dtype=np.float32)

    out[:, :, :256] = (
        cube[:, :, :256] - stats["eps_mean"]
    ) / stats["eps_std"]

    out[:, :, 256] = (
        cube[:, :, 256] - stats["temp_mean"]
    ) / stats["temp_std"]

    out[:, :, 257] = (
        cube[:, :, 257] - stats["depth_mean"]
    ) / stats["depth_std"]

    return out


# ================================================================
# PATCH DATASET
# ================================================================

def patch_coordinates(
    height,
    x_start,
    x_end,
    patch_size,
    stride,
):
    coords = []

    for y in range(0, height - patch_size + 1, stride):
        for x in range(
            x_start,
            x_end - patch_size + 1,
            stride,
        ):
            coords.append((y, x))

    return coords


class PhysicalPatchDataset(Dataset):

    def __init__(self, cube, coordinates, patch_size=32):
        # Guardamos CHW una sola vez.
        self.cube = (
            torch.from_numpy(cube)
            .permute(2, 0, 1)
            .contiguous()
        )
        self.coordinates = coordinates
        self.patch_size = patch_size

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        y, x = self.coordinates[idx]

        return self.cube[
            :,
            y:y + self.patch_size,
            x:x + self.patch_size,
        ]


def create_loaders(cube, regions, args):
    train_coords = patch_coordinates(
        H_EXPECTED,
        *regions["train"],
        args.patch_size,
        args.train_stride,
    )

    val_coords = patch_coordinates(
        H_EXPECTED,
        *regions["val"],
        args.patch_size,
        args.val_stride,
    )

    test_coords = patch_coordinates(
        H_EXPECTED,
        *regions["test"],
        args.patch_size,
        args.test_stride,
    )

    train_ds = PhysicalPatchDataset(
        cube, train_coords, args.patch_size
    )
    val_ds = PhysicalPatchDataset(
        cube, val_coords, args.patch_size
    )
    test_ds = PhysicalPatchDataset(
        cube, test_coords, args.patch_size
    )

    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    if args.num_workers > 0:
        loader_args["persistent_workers"] = True

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        generator=generator,
        **loader_args,
    )

    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **loader_args,
    )

    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        **loader_args,
    )

    print("\n========== PATCHES ==========")
    print("Train patches:", len(train_ds))
    print("Val patches  :", len(val_ds))
    print("Test patches :", len(test_ds))

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


# ================================================================
# RED 1 - ENCODER (MULTIMODAL)
# ================================================================

class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ModalityEncoder(nn.Module):
    """
    Encoder propio de UNA modalidad:

        in_channels x 32 x 32
              ↓
        mid_channels x 16 x 16   (skip a 16x16)
              ↓
        out_channels x 8 x 8     (skip a 8x8)
    """

    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()

        self.block1 = EncoderBlock(in_channels, mid_channels)
        self.block2 = EncoderBlock(mid_channels, out_channels)

    def forward(self, x):
        feat16 = self.block1(x)
        feat8 = self.block2(feat16)
        return feat16, feat8


class PhysicalEncoder(nn.Module):
    """
    Emisividad  (256x32x32) -> enc_eps   -> 64 x 8 x 8 ─┐
    Temperatura (1x32x32)   -> enc_temp  -> 32 x 8 x 8 ─┼─ concat
    Profundidad (1x32x32)   -> enc_depth -> 32 x 8 x 8 ─┘   128 x 8 x 8
                                                              ↓
                                                     encoder de fusión
                                                        128 x 4 x 4
                                                              ↓
                                                        2048 features
                                                              ↓
                                                         latent_dim

    Cada modalidad tiene su propio encoder y luego un encoder
    de fusión las junta en UN solo vector latente.

    2048 sigue siendo la dimensión máxima natural del cuello de
    botella de esta arquitectura y podemos barrer hacia abajo.

    forward devuelve (z, skips). skips solo lo usa el decoder
    cuando se activan las skip connections.
    """

    def __init__(self, latent_dim):
        super().__init__()

        if latent_dim < 1 or latent_dim > 2048:
            raise ValueError(
                "latent_dim debe estar entre 1 y 2048."
            )

        self.eps_encoder = ModalityEncoder(N_EMISSIVITY, 128, 64)
        self.temp_encoder = ModalityEncoder(1, 16, 32)
        self.depth_encoder = ModalityEncoder(1, 16, 32)

        # 64 + 32 + 32 = 128 canales a 8x8 -> 128 x 4 x 4
        self.fusion_encoder = EncoderBlock(128, 128)

        self.flatten = nn.Flatten()

        self.fc = nn.Linear(
            128 * 4 * 4,
            latent_dim,
        )

    def forward(self, x):
        eps = x[:, :N_EMISSIVITY]
        temp = x[:, N_EMISSIVITY:N_EMISSIVITY + 1]
        depth = x[:, N_EMISSIVITY + 1:N_PHYSICAL]

        eps16, eps8 = self.eps_encoder(eps)
        temp16, temp8 = self.temp_encoder(temp)
        depth16, depth8 = self.depth_encoder(depth)

        fused = torch.cat([eps8, temp8, depth8], dim=1)
        fused = self.fusion_encoder(fused)

        z = self.fc(self.flatten(fused))

        skips = {
            "eps": (eps16, eps8),
            "temp": (temp16, temp8),
            "depth": (depth16, depth8),
        }

        return z, skips


# ================================================================
# RED 1 - DECODER (3 CABEZAS)
# ================================================================

class DecoderBlock(nn.Module):
    """
    En vez de ConvTranspose2d usamos:
        Upsample -> ReflectionPad2d -> Conv2d

    La reflexión evita introducir ceros artificiales en los bordes.
    """

    def __init__(self, in_channels, out_channels, final=False):
        super().__init__()

        layers = [
            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            ),
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=0,
                bias=final,
            ),
        ]

        if not final:
            layers += [
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DecoderHead(nn.Module):
    """
    Cabeza de UNA modalidad:

        128 x 8 x 8 (tronco compartido)
              ↓  [+ skip 8x8 si use_skips]
        mid_channels x 16 x 16
              ↓  [+ skip 16x16 si use_skips]
        out_channels x 32 x 32
    """

    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        skip8_channels,
        skip16_channels,
        use_skips,
    ):
        super().__init__()

        self.use_skips = use_skips

        extra8 = skip8_channels if use_skips else 0
        extra16 = skip16_channels if use_skips else 0

        self.up1 = DecoderBlock(in_channels + extra8, mid_channels)
        self.up2 = DecoderBlock(
            mid_channels + extra16,
            out_channels,
            final=True,
        )

    def forward(self, x, skip8=None, skip16=None):
        if self.use_skips:
            x = torch.cat([x, skip8], dim=1)

        x = self.up1(x)

        if self.use_skips:
            x = torch.cat([x, skip16], dim=1)

        return self.up2(x)


class PhysicalDecoder(nn.Module):
    """
    latent_dim
        ↓
    128 x 4 x 4
        ↓  tronco compartido
    128 x 8 x 8
        ├── cabeza emisividad  -> 256 x 32 x 32
        ├── cabeza temperatura ->   1 x 32 x 32
        └── cabeza profundidad ->   1 x 32 x 32

    La salida se concatena en 258 x 32 x 32 (mismo formato de antes).
    """

    def __init__(self, latent_dim, use_skips=False):
        super().__init__()

        self.use_skips = use_skips

        self.fc = nn.Linear(
            latent_dim,
            128 * 4 * 4,
        )

        self.trunk = DecoderBlock(128, 128)

        self.eps_head = DecoderHead(
            128, 256, N_EMISSIVITY, 64, 128, use_skips
        )
        self.temp_head = DecoderHead(
            128, 64, 1, 32, 16, use_skips
        )
        self.depth_head = DecoderHead(
            128, 64, 1, 32, 16, use_skips
        )

    def forward(self, z, skips=None):
        x = self.fc(z)
        x = x.view(-1, 128, 4, 4)
        x = self.trunk(x)

        if self.use_skips:
            if skips is None:
                raise ValueError(
                    "El decoder tiene skip connections "
                    "y necesita los skips del encoder."
                )
            eps16, eps8 = skips["eps"]
            temp16, temp8 = skips["temp"]
            depth16, depth8 = skips["depth"]
        else:
            eps16 = eps8 = None
            temp16 = temp8 = None
            depth16 = depth8 = None

        eps = self.eps_head(x, eps8, eps16)
        temp = self.temp_head(x, temp8, temp16)
        depth = self.depth_head(x, depth8, depth16)

        return torch.cat([eps, temp, depth], dim=1)


# ================================================================
# AUTOENCODER COMPLETO
# ================================================================

class PhysicalAutoencoder(nn.Module):

    def __init__(self, latent_dim, use_skips=False):
        super().__init__()

        self.encoder = PhysicalEncoder(latent_dim)
        self.decoder = PhysicalDecoder(latent_dim, use_skips)

    def forward(self, x):
        z, skips = self.encoder(x)
        reconstruction = self.decoder(z, skips)
        return reconstruction, z

# ================================================================
# PÉRDIDA PONDERADA POR PIXEL
# ================================================================

def pixel_weight_map(target, alpha):
    """
    Peso por pixel calculado con el ground truth (B, C, H, W).
    Pixeles en bordes o picos (gradiente alto) pesan más que los
    de zonas planas. Devuelve (B, 1, H, W) con media 1.
    """
    with torch.no_grad():
        dx = (target[:, :, :, 1:] - target[:, :, :, :-1]).abs()
        dy = (target[:, :, 1:, :] - target[:, :, :-1, :]).abs()

        # Cada pixel recibe la diferencia con sus dos vecinos.
        gx = F.pad(dx, (0, 1, 0, 0)) + F.pad(dx, (1, 0, 0, 0))
        gy = F.pad(dy, (0, 0, 0, 1)) + F.pad(dy, (0, 0, 1, 0))

        # Un solo mapa por pixel (promedio sobre canales).
        g = (gx + gy).mean(dim=1, keepdim=True)
        g = g / (g.mean(dim=(2, 3), keepdim=True) + 1e-6)
        g = g.clamp(max=10.0)   # evita que un pico domine todo

        w = 1.0 + alpha * g
        w = w / w.mean(dim=(2, 3), keepdim=True)

    return w


def pixel_weighted_mse(pred, true, alpha):
    w = pixel_weight_map(true, alpha)
    return (w * (pred - true) ** 2).mean()

# ================================================================
# LIGHTNING MODULE
# ================================================================

class PhysicalAutoencoderLightning(L.LightningModule):

    def __init__(
        self,
        latent_dim,
        lr=1e-3,
        weight_eps=1.0,
        weight_temp=1.0,
        weight_depth=1.0,
        use_skips=False,
        edge_alpha=1.0,
        max_epochs=120,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.encoder = PhysicalEncoder(latent_dim)
        self.decoder = PhysicalDecoder(latent_dim, use_skips)

    def forward(self, x):
        z, skips = self.encoder(x)
        reconstruction = self.decoder(z, skips)
        return reconstruction, z

    def calculate_losses(self, batch, return_reconstruction=False):
        reconstruction, z = self(batch)

        eps_true = batch[:, :256]
        temp_true = batch[:, 256:257]
        depth_true = batch[:, 257:258]

        eps_pred = reconstruction[:, :256]
        temp_pred = reconstruction[:, 256:257]
        depth_pred = reconstruction[:, 257:258]

        alpha = self.hparams.edge_alpha

        loss_eps = pixel_weighted_mse(eps_pred, eps_true, alpha)
        loss_temp = pixel_weighted_mse(temp_pred, temp_true, alpha)
        loss_depth = pixel_weighted_mse(depth_pred, depth_true, alpha)

        loss_total = (
            self.hparams.weight_eps * loss_eps
            + self.hparams.weight_temp * loss_temp
            + self.hparams.weight_depth * loss_depth
        )

        if return_reconstruction:
            return (
                loss_total,
                loss_eps,
                loss_temp,
                loss_depth,
                reconstruction,
            )

        return (
            loss_total,
            loss_eps,
            loss_temp,
            loss_depth,
        )

    def training_step(self, batch, batch_idx):
        total, eps, temp, depth = self.calculate_losses(batch)

        self.log(
            "train_total",
            total,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_eps",
            eps,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_temp",
            temp,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_depth",
            depth,
            on_step=False,
            on_epoch=True,
        )

        return total

    def validation_step(self, batch, batch_idx):
        (
            total,
            eps,
            temp,
            depth,
            reconstruction,
        ) = self.calculate_losses(batch, return_reconstruction=True)

        self.log(
            "val_total",
            total,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_eps",
            eps,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_temp",
            temp,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_depth",
            depth,
            on_step=False,
            on_epoch=True,
        )

        # ------------------------------------------------------------
        # Métricas para early stopping / mejor checkpoint.
        # Se promedian por igual las 3 modalidades para que las 256
        # bandas de emisividad no ahoguen temperatura y profundidad.
        # ------------------------------------------------------------
        mae_eps = F.l1_loss(reconstruction[:, :256], batch[:, :256])
        mae_temp = F.l1_loss(
            reconstruction[:, 256:257], batch[:, 256:257]
        )
        mae_depth = F.l1_loss(
            reconstruction[:, 257:258], batch[:, 257:258]
        )

        val_mae = (mae_eps + mae_temp + mae_depth) / 3.0
        mse_eps = F.mse_loss(reconstruction[:, :256], batch[:, :256])
        mse_temp = F.mse_loss(
            reconstruction[:, 256:257], batch[:, 256:257]
        )
        mse_depth = F.mse_loss(
            reconstruction[:, 257:258], batch[:, 257:258]
        )
        val_mse = (mse_eps + mse_temp + mse_depth) / 3.0

        self.log(
            "val_mae",
            val_mae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_mse",
            val_mse,
            on_step=False,
            on_epoch=True,
        )

        return total

    def test_step(self, batch, batch_idx):
        total, eps, temp, depth = self.calculate_losses(batch)

        self.log("test_total", total, on_step=False, on_epoch=True)
        self.log("test_eps", eps, on_step=False, on_epoch=True)
        self.log("test_temp", temp, on_step=False, on_epoch=True)
        self.log("test_depth", depth, on_step=False, on_epoch=True)

        return total

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
        )

        # Scheduler de coseno: el LR decae suavemente desde --lr
        # hasta ~0 a lo largo de las --max-epochs. Se actualiza una
        # vez por época (interval="epoch"), así que para que el
        # coseno complete su recorrido el entrenamiento no debe
        # cortarse por early stopping mucho antes de max_epochs
        # (por eso subimos --patience por defecto).
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


# ================================================================
# MÉTRICAS EN UNIDADES FÍSICAS
# ================================================================

@torch.no_grad()
def physical_metrics(model, loader, device, stats):

    model.eval()
    model.to(device)

    eps_mean = torch.tensor(
        stats["eps_mean"],
        dtype=torch.float32,
        device=device,
    ).reshape(1, 256, 1, 1)

    eps_std = torch.tensor(
        stats["eps_std"],
        dtype=torch.float32,
        device=device,
    ).reshape(1, 256, 1, 1)

    temp_mean = torch.tensor(
        stats["temp_mean"],
        dtype=torch.float32,
        device=device,
    )
    temp_std = torch.tensor(
        stats["temp_std"],
        dtype=torch.float32,
        device=device,
    )

    depth_mean = torch.tensor(
        stats["depth_mean"],
        dtype=torch.float32,
        device=device,
    )
    depth_std = torch.tensor(
        stats["depth_std"],
        dtype=torch.float32,
        device=device,
    )

    accum = {
        "eps_sq": 0.0,
        "eps_abs": 0.0,
        "eps_n": 0,
        "temp_sq": 0.0,
        "temp_abs": 0.0,
        "temp_n": 0,
        "depth_sq": 0.0,
        "depth_abs": 0.0,
        "depth_n": 0,
    }

    for batch in loader:

        batch = batch.to(device, non_blocking=True)

        prediction, _ = model(batch)

        eps_true = (
            batch[:, :256] * eps_std
            + eps_mean
        )
        eps_pred = (
            prediction[:, :256] * eps_std
            + eps_mean
        )

        temp_true = (
            batch[:, 256:257] * temp_std
            + temp_mean
        )
        temp_pred = (
            prediction[:, 256:257] * temp_std
            + temp_mean
        )

        depth_true = (
            batch[:, 257:258] * depth_std
            + depth_mean
        )
        depth_pred = (
            prediction[:, 257:258] * depth_std
            + depth_mean
        )

        e = eps_pred - eps_true
        t = temp_pred - temp_true
        d = depth_pred - depth_true

        accum["eps_sq"] += torch.sum(e ** 2).item()
        accum["eps_abs"] += torch.sum(torch.abs(e)).item()
        accum["eps_n"] += e.numel()

        accum["temp_sq"] += torch.sum(t ** 2).item()
        accum["temp_abs"] += torch.sum(torch.abs(t)).item()
        accum["temp_n"] += t.numel()

        accum["depth_sq"] += torch.sum(d ** 2).item()
        accum["depth_abs"] += torch.sum(torch.abs(d)).item()
        accum["depth_n"] += d.numel()

    return {
        "eps_rmse": math.sqrt(
            accum["eps_sq"] / accum["eps_n"]
        ),
        "eps_mae": (
            accum["eps_abs"] / accum["eps_n"]
        ),
        "temp_rmse": math.sqrt(
            accum["temp_sq"] / accum["temp_n"]
        ),
        "temp_mae": (
            accum["temp_abs"] / accum["temp_n"]
        ),
        "depth_rmse": math.sqrt(
            accum["depth_sq"] / accum["depth_n"]
        ),
        "depth_mae": (
            accum["depth_abs"] / accum["depth_n"]
        ),
    }


# ================================================================
# VALIDACIÓN VISUAL
# ================================================================

def denormalize_patch(patch, stats):

    eps = (
        patch[:256]
        * stats["eps_std"].reshape(256, 1, 1)
        + stats["eps_mean"].reshape(256, 1, 1)
    )

    eps = np.clip(eps, 0.0, 1.0)

    temp = (
        patch[256]
        * stats["temp_std"]
        + stats["temp_mean"]
        - 273.15
    )

    depth = (
        patch[257]
        * stats["depth_std"]
        + stats["depth_mean"]
    )

    return eps, temp, depth


@torch.no_grad()
def save_visual_validation(
    model,
    dataset,
    patch_index,
    device,
    stats,
    output_dir,
    latent_dim,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    if patch_index >= len(dataset):
        raise IndexError(
            f"El parche {patch_index} no existe. "
            f"Hay {len(dataset)} parches."
        )

    model.eval()
    model.to(device)

    sample = dataset[patch_index].unsqueeze(0).to(device)

    reconstruction, z = model(sample)

    original = sample[0].cpu().numpy()
    predicted = reconstruction[0].cpu().numpy()

    eps_o, temp_o, depth_o = denormalize_patch(
        original,
        stats,
    )

    eps_p, temp_p, depth_p = denormalize_patch(
        predicted,
        stats,
    )

    # ------------------------------------------------------------
    # Temperatura y profundidad
    # ------------------------------------------------------------

    for truth, pred, title, unit, filename, cmap in [
        (
            temp_o,
            temp_p,
            "Temperatura",
            "°C",
            "temperature.png",
            CMAP_TEMPERATURE,
        ),
        (
            depth_o,
            depth_p,
            "Profundidad",
            "m",
            "depth.png",
            CMAP_DEPTH,
        ),
    ]:
        error = np.abs(truth - pred)

        lo = min(float(truth.min()), float(pred.min()))
        hi = max(float(truth.max()), float(pred.max()))

        if hi == lo:
            hi = lo + 1e-6

        fig, ax = plt.subplots(
            1,
            3,
            figsize=(13, 4),
        )

        im = ax[0].imshow(
            truth,
            vmin=lo,
            vmax=hi,
            cmap=cmap,
        )
        ax[0].set_title(f"{title} original")
        ax[0].axis("off")
        fig.colorbar(im, ax=ax[0], label=unit)

        im = ax[1].imshow(
            pred,
            vmin=lo,
            vmax=hi,
            cmap=cmap,
        )
        ax[1].set_title(f"{title} reconstruida")
        ax[1].axis("off")
        fig.colorbar(im, ax=ax[1], label=unit)

        im = ax[2].imshow(error)
        ax[2].set_title(
            f"Error absoluto\n"
            f"RMSE={np.sqrt(np.mean((truth-pred)**2)):.5g}"
        )
        ax[2].axis("off")
        fig.colorbar(im, ax=ax[2], label=unit)

        fig.suptitle(
            f"z={latent_dim} | parche validation {patch_index}"
        )
        fig.tight_layout()

        fig.savefig(
            output_dir / filename,
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)

    # ------------------------------------------------------------
    # Emisividad: varias bandas
    # ------------------------------------------------------------

    bands = [0]

    fig, axes = plt.subplots(
        len(bands),
        3,
        figsize=(11, 3 * len(bands)),
        squeeze=False,
    )

    for row, band in enumerate(bands):

        truth = eps_o[band]
        pred = eps_p[band]
        error = np.abs(truth - pred)

        lo = min(float(truth.min()), float(pred.min()))
        hi = max(float(truth.max()), float(pred.max()))

        if hi == lo:
            hi = lo + 1e-6

        axes[row, 0].imshow(
            truth,
            vmin=lo,
            vmax=hi,
            cmap=CMAP_EMISSIVITY,
        )
        axes[row, 0].set_title(
            f"Banda {band} - Original"
        )

        axes[row, 1].imshow(
            pred,
            vmin=lo,
            vmax=hi,
            cmap=CMAP_EMISSIVITY,
        )
        axes[row, 1].set_title(
            f"Banda {band} - Reconstruida"
        )

        axes[row, 2].imshow(error)
        axes[row, 2].set_title(
            f"Banda {band} - Error"
        )

        for col in range(3):
            axes[row, col].axis("off")

    fig.suptitle(
        f"Emisividad | z={latent_dim}"
    )
    fig.tight_layout()

    fig.savefig(
        output_dir / "emissivity_bands.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------
    # Firma espectral del píxel central
    # ------------------------------------------------------------

    py = eps_o.shape[1] // 2
    px = eps_o.shape[2] // 2

    fig = plt.figure(figsize=(9, 5))

    plt.plot(
        eps_o[:, py, px],
        label="Original",
    )

    plt.plot(
        eps_p[:, py, px],
        label="Reconstruida",
    )

    plt.xlabel("Banda espectral")
    plt.ylabel("Emisividad")
    plt.title(
        f"Firma espectral | z={latent_dim}"
    )
    plt.grid()
    plt.legend()
    plt.tight_layout()

    fig.savefig(
        output_dir / "emissivity_spectrum.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Guardamos también el vector latente.
    np.save(
        output_dir / "latent_vector.npy",
        z[0].cpu().numpy(),
    )
    
# ================================================================
# RECONSTRUCCIÓN DE LA ESCENA COMPLETA
# ================================================================

@torch.no_grad()
def save_full_scene_reconstruction(
    model,
    normalized_cube,
    physical_cube,
    regions,
    device,
    stats,
    output_dir,
    latent_dim,
    patch_size=32,
    stride=16,
    bands=(0, 128, 255),
    batch_size=64,
):
    """
    Reconstruye TODA la escena parche por parche y guarda una imagen
    con original / reconstruida / error para temperatura, profundidad
    y algunas bandas de emisividad.

    Los parches se solapan (stride < patch_size) y se promedian en
    los traslapes. Cada región (train/val/test) se procesa por
    separado, así ningún parche mezcla regiones.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.to(device)

    H, W, _ = normalized_cube.shape

    # ------------------------------------------------------------
    # Ventana Gaussiana para blending de parches
    # ------------------------------------------------------------
    sigma = patch_size / 4.0

    gaussian_coords = (
        np.arange(patch_size)
        - (patch_size - 1) / 2
    )

    gaussian_1d = np.exp(
        -(gaussian_coords ** 2)
        / (2 * sigma ** 2)
    )

    gaussian = np.outer(
        gaussian_1d,
        gaussian_1d,
    ).astype(np.float32)

    gaussian /= gaussian.max()

    # Canales que se dibujan: bandas de emisividad + temp + depth.
    channels = list(range(N_PHYSICAL))

    acc = np.zeros((H, W, len(channels)), dtype=np.float32)
    cnt = np.zeros((H, W, 1), dtype=np.float32)

    def starts(start, end):
        s = list(range(start, end - patch_size + 1, stride))
        if s[-1] != end - patch_size:
            s.append(end - patch_size)   # último parche pegado al borde
        return s

    for x0, x1 in regions.values():
        coords = [
            (y, x)
            for y in starts(0, H)
            for x in starts(x0, x1)
        ]

        for i in range(0, len(coords), batch_size):
            chunk = coords[i:i + batch_size]

            batch = torch.stack([
                torch.from_numpy(
                    normalized_cube[
                        y:y + patch_size,
                        x:x + patch_size,
                        :,
                    ]
                ).permute(2, 0, 1)
                for y, x in chunk
            ]).to(device)

            pred, _ = model(batch)
            pred = (
                pred[:, channels]
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
            )

            for (y, x), p in zip(chunk, pred):

                acc[
                    y:y + patch_size,
                    x:x + patch_size
                ] += p * gaussian[:, :, None]

                cnt[
                    y:y + patch_size,
                    x:x + patch_size
                ] += gaussian[:, :, None]

    recon = acc / np.maximum(cnt, 1e-8)

    # ------------------------------------------------------------
    # Mapas predichos completos en unidades físicas (.npy)
    # ------------------------------------------------------------
    eps_pred_full = np.clip(
        recon[:, :, :N_EMISSIVITY]
        * stats["eps_std"].reshape(1, 1, N_EMISSIVITY)
        + stats["eps_mean"].reshape(1, 1, N_EMISSIVITY),
        0.0,
        1.0,
    ).astype(np.float32)

    temp_pred_full = (
        recon[:, :, N_EMISSIVITY] * stats["temp_std"] + stats["temp_mean"]
    ).astype(np.float32)          # Kelvin, igual que el .npy de entrada

    depth_pred_full = (
        recon[:, :, N_EMISSIVITY + 1] * stats["depth_std"]
        + stats["depth_mean"]
    ).astype(np.float32)

    np.save(output_dir / "predicted_emissivity_map.npy", eps_pred_full)
    np.save(output_dir / "predicted_temperature.npy", temp_pred_full)
    np.save(output_dir / "predicted_depth.npy", depth_pred_full)

    del eps_pred_full, temp_pred_full, depth_pred_full

    def denormalize(values, channel):

        if channel < N_EMISSIVITY:

            values = (
                values * stats["eps_std"][0, 0, channel]
                + stats["eps_mean"][0, 0, channel]
            )

            return np.clip(values, 0.0, 1.0)

        if channel == N_EMISSIVITY:

            temperature_K = (
                values * stats["temp_std"]
                + stats["temp_mean"]
            )

            return temperature_K - 273.15

        return (
            values * stats["depth_std"]
            + stats["depth_mean"]
        )

    # Orden de dibujo: temperatura, profundidad, bandas de emisividad.
    n_bands = len(bands)
    order = [
        (N_EMISSIVITY, "Temperatura", "°C"),
        (N_EMISSIVITY + 1, "Profundidad", "m"),
    ] + [
        (b, f"Emisividad banda {b}", "")
        for b in bands
    ]

    n_rows = 3 * len(order)

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(18, 3.0 * n_rows),
    )

    row = 0

    for k, title, unit in order:
        channel = channels[k]

        if channel == 256:
            truth = physical_cube[:, :, channel] - 273.15
        else:
            truth = physical_cube[:, :, channel]
        pred = denormalize(recon[:, :, k], channel)
        error = np.abs(truth - pred)

        lo = min(float(truth.min()), float(pred.min()))
        hi = max(float(truth.max()), float(pred.max()))

        if hi == lo:
            hi = lo + 1e-6

        if title.startswith("Temperatura"):
            cmap = CMAP_TEMPERATURE
        elif title.startswith("Profundidad"):
            cmap = CMAP_DEPTH
        else:
            cmap = CMAP_EMISSIVITY

        panels = [
            (truth, f"{title} - Original", lo, hi, cmap),
            (pred, f"{title} - Reconstruida", lo, hi, cmap),
            (
                error,
                f"{title} - Error absoluto (MAE={error.mean():.5g})",
                None,
                None,
                None,
            ),
        ]

        for image, panel_title, vmin, vmax, panel_cmap in panels:
            ax = axes[row]

            im = ax.imshow(
                image,
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
                cmap=panel_cmap,
            )
            ax.set_title(panel_title, fontsize=11)
            ax.axis("off")
            fig.colorbar(
                im,
                ax=ax,
                fraction=0.02,
                pad=0.01,
                label=unit,
            )

            # Límites train / val / test.
            for name, (x0, x1) in regions.items():
                if x0 > 0:
                    ax.axvline(
                        x0 - 0.5,
                        color="white",
                        linestyle="--",
                        linewidth=1,
                    )
                if row == 0:
                    ax.text(
                        (x0 + x1) / 2,
                        6,
                        name,
                        color="white",
                        ha="center",
                        va="top",
                        fontsize=12,
                        bbox=dict(
                            facecolor="black",
                            alpha=0.5,
                            pad=2,
                        ),
                    )

            row += 1

    fig.suptitle(
        f"Escena completa reconstruida | z={latent_dim}",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    fig.savefig(
        output_dir / "scene_reconstruction.png",
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)

# ================================================================
# SELECCIÓN DEL LATENTE
# ================================================================

def select_latent(summary, tolerance):

    # El mayor latente es la referencia.
    baseline_row = summary.loc[
        summary["latent_dim"].idxmax()
    ]

    metrics = [
        "val_total",
        "eps_rmse",
        "temp_rmse",
        "depth_rmse",
    ]

    mask = np.ones(len(summary), dtype=bool)

    for metric in metrics:

        limit = (
            baseline_row[metric]
            * (1.0 + tolerance)
        )

        mask &= (
            summary[metric].to_numpy()
            <= limit
        )

    candidates = summary.loc[mask].sort_values(
        "latent_dim"
    )

    if len(candidates) == 0:
        return int(baseline_row["latent_dim"])

    return int(
        candidates.iloc[0]["latent_dim"]
    )


# ================================================================
# MAIN
# ================================================================

def main():

    args = parse_args()

    L.seed_everything(
        args.seed,
        workers=True,
    )

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # 1. Cargar
    # ------------------------------------------------------------

    data_dir = ensure_extracted_data(data_dir, args.drive_folder_id)
    physical_cube = load_data(data_dir)

    regions = spatial_regions(
        W_EXPECTED,
        args.train_fraction,
        args.val_fraction,
    )

    print("\n========== REGIONES ==========")

    for name, (x0, x1) in regions.items():
        print(
            f"{name:5s}: {x0} -> {x1} "
            f"(ancho={x1-x0})"
        )

    # ------------------------------------------------------------
    # 2. Normalización SOLO usando TRAIN
    # ------------------------------------------------------------

    stats = compute_normalization(
        physical_cube,
        regions,
    )

    normalized_cube = normalize_cube(
        physical_cube,
        stats,
    )

    np.savez(
        output_dir / "normalization_stats.npz",
        eps_mean=stats["eps_mean"],
        eps_std=stats["eps_std"],
        temp_mean=stats["temp_mean"],
        temp_std=stats["temp_std"],
        depth_mean=stats["depth_mean"],
        depth_std=stats["depth_std"],
    )

    # ------------------------------------------------------------
    # 3. Datasets
    # ------------------------------------------------------------

    (
        train_ds,
        val_ds,
        test_ds,
        train_loader,
        val_loader,
        test_loader,
    ) = create_loaders(
        normalized_cube,
        regions,
        args,
    )

    # Verificación de dimensiones.
    batch = next(iter(train_loader))

    print("\n========== VERIFICACIÓN ==========")
    print("Batch:", tuple(batch.shape))
    print("Skip connections:", args.skip_connections)
    print("Métrica early stopping / checkpoint:", args.monitor)

    # ------------------------------------------------------------
    # 4. Barrido
    # ------------------------------------------------------------

    latent_dims = sorted(
        set(args.latent_dims),
        reverse=True,
    )

    results = []

    for latent_dim in latent_dims:

        print("\n")
        print("=" * 70)
        print(
            f"ENTRENAMIENTO: LATENT_DIM = {latent_dim}"
        )
        print("=" * 70)

        L.seed_everything(
            args.seed,
            workers=True,
        )

        exp_dir = (
            output_dir
            / f"latent_{latent_dim}"
        )

        exp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.rmtree(exp_dir / "checkpoints", ignore_errors=True)
        shutil.rmtree(exp_dir / "logs", ignore_errors=True)

        checkpoint = ModelCheckpoint(
            dirpath=exp_dir / "checkpoints",
            filename="best",
            monitor=args.monitor,
            mode="min",
            save_top_k=1,
        )

        early_stopping = EarlyStopping(
            monitor=args.monitor,
            mode="min",
            patience=args.patience,
            verbose=True,
        )

        logger = CSVLogger(
            save_dir=str(exp_dir),
            name="logs",
        )

        model = PhysicalAutoencoderLightning(
            latent_dim=latent_dim,
            lr=args.lr,
            weight_eps=args.weight_eps,
            weight_temp=args.weight_temp,
            weight_depth=args.weight_depth,
            use_skips=args.skip_connections,
            edge_alpha=args.edge_alpha,
            max_epochs=args.max_epochs,
        )

        precision = (
            "16-mixed"
            if torch.cuda.is_available()
            else "32-true"
        )

        trainer = L.Trainer(
            max_epochs=args.max_epochs,
            accelerator="auto",
            devices=1,
            precision=precision,
            callbacks=[
                checkpoint,
                early_stopping,
            ],
            logger=logger,
            log_every_n_steps=10,
            enable_progress_bar=True,
            benchmark=torch.cuda.is_available(),
        )

        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

        best_path = checkpoint.best_model_path

        if not best_path:
            raise RuntimeError(
                f"No se generó checkpoint para "
                f"z={latent_dim}"
            )

        best_model = (
            PhysicalAutoencoderLightning
            .load_from_checkpoint(best_path)
        )

        val_result = trainer.validate(
            best_model,
            dataloaders=val_loader,
            verbose=False,
        )[0]

        test_result = trainer.test(
            best_model,
            dataloaders=test_loader,
            verbose=False,
        )[0]

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        physical = physical_metrics(
            best_model,
            test_loader,
            device,
            stats,
        )

        visual_dir = (
            exp_dir
            / "visual_validation"
        )

        save_visual_validation(
            best_model,
            val_ds,
            args.visual_patch_index,
            device,
            stats,
            visual_dir,
            latent_dim,
        )

        save_full_scene_reconstruction(
            best_model,
            normalized_cube,
            physical_cube,
            regions,
            device,
            stats,
            visual_dir,
            latent_dim,
            patch_size=args.patch_size,
            stride=args.scene_stride,
            bands=args.scene_bands,
        )

        row = {
            "latent_dim": latent_dim,
            "val_total": float(
                val_result["val_total"]
            ),
            "val_eps": float(
                val_result["val_eps"]
            ),
            "val_temp": float(
                val_result["val_temp"]
            ),
            "val_depth": float(
                val_result["val_depth"]
            ),
            "test_total": float(
                test_result["test_total"]
            ),
            "test_eps": float(
                test_result["test_eps"]
            ),
            "test_temp": float(
                test_result["test_temp"]
            ),
            "test_depth": float(
                test_result["test_depth"]
            ),
            **physical,
            "checkpoint": best_path,
        }

        results.append(row)

        partial = pd.DataFrame(results).sort_values(
            "latent_dim",
            ascending=False,
        )

        partial.to_csv(
            output_dir
            / "sweep_summary_partial.csv",
            index=False,
        )

        print("\n========== RESULTADO ==========")

        for key, value in row.items():
            if key != "checkpoint":
                print(
                    f"{key}: {value}"
                )

        # Liberar memoria antes del siguiente experimento.
        del trainer
        del model
        del best_model

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # 5. Resumen final
    # ------------------------------------------------------------

    summary = pd.DataFrame(results).sort_values(
        "latent_dim",
        ascending=False,
    )

    summary.to_csv(
        output_dir / "sweep_summary.csv",
        index=False,
    )

    selected_latent = select_latent(
        summary,
        args.tolerance,
    )

    selected_row = summary[
        summary["latent_dim"]
        == selected_latent
    ].iloc[0]

    # Guardar información de selección.
    selection = {
        "selected_latent_dim": selected_latent,
        "baseline_latent_dim": int(
            summary["latent_dim"].max()
        ),
        "tolerance": args.tolerance,
        "rule": (
            "Menor latente cuyo val_total, eps_rmse, "
            "temp_rmse y depth_rmse no empeoran más "
            "que la tolerancia frente al modelo de "
            "mayor dimensión."
        ),
        "selected_checkpoint": selected_row[
            "checkpoint"
        ],
    }

    with open(
        output_dir / "selection.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            selection,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------
    # 6. Guardar decoder/encoder del seleccionado
    # ------------------------------------------------------------

    selected_model = (
        PhysicalAutoencoderLightning
        .load_from_checkpoint(
            selected_row["checkpoint"]
        )
    )

    selected_artifact = {
        "latent_dim": selected_latent,
        "skip_connections": args.skip_connections,
        "encoder_state_dict":
            selected_model.encoder.state_dict(),
        "decoder_state_dict":
            selected_model.decoder.state_dict(),
        "model_state_dict":
            selected_model.state_dict(),
        "normalization": stats,
        "input_shape": (
            258,
            args.patch_size,
            args.patch_size,
        ),
        "image_shape": (
            H_EXPECTED,
            W_EXPECTED,
        ),
    }

    torch.save(
        selected_artifact,
        output_dir
        / f"selected_physical_autoencoder_z{selected_latent}.pt",
    )

    print("\n")
    print("=" * 70)
    print("BARRIDO TERMINADO")
    print("=" * 70)
    print(summary.to_string(index=False))

    print(
        f"\nSelección preliminar: z={selected_latent}"
    )

    print(
        "\nIMPORTANTE: antes de utilizar este decoder "
        "en la Red 2, revise las imágenes de "
        "'visual_validation' de todos los latentes."
    )


if __name__ == "__main__":
    main()