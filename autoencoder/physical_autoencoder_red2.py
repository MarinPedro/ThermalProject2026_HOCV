#!/usr/bin/env python3
# ================================================================
# PHYSICAL AUTOENCODER - RED 2
#
# Idea:
#   simulated_hsi (256 bandas) -> encoders multimodales -> z
#   z -> DECODER DE LA RED 1 (congelado) -> emisividad,
#                                           temperatura y profundidad
#
# - Input: cubo simulado simulated_hsi, forma (260, 1560, 256).
# - Tres encoders (emisividad, temperatura, profundidad). Los tres
#   reciben EXACTAMENTE el mismo cubo simulado con todas sus bandas,
#   y cada uno tiene sus propios pesos.
# - Un encoder de fusión los une en UN solo latente (misma estructura
#   que la Red 1).
# - El decoder se carga del checkpoint de la Red 1 y queda CONGELADO:
#   solo cambian los pesos de los encoders.
# - Si la Red 1 se entrenó CON skip connections, el decoder las espera:
#   en ese caso los encoders de la Red 2 producen también los skips
#   (features a 16x16 y 8x8 de cada modalidad, calculadas desde el HSI
#   simulado, sin usar los ground truths). Si se entrenó sin skips, el
#   decoder recibe solo z. El modo se detecta solo, desde el checkpoint.
# - Targets: los mismos mapas físicos de la Red 1 (258 canales),
#   normalizados con las MISMAS estadísticas de la Red 1.
# - Pérdida: solo sobre los mapas (MSE ponderado por pixel, igual que
#   la Red 1), más regularización de Tikhonov sobre el eje espectral de
#   la emisividad (--tikhonov-alpha). Early stopping y mejor checkpoint
#   con val_mae / val_mse.
# - Mismo split espacial 70/15/15 y parches de 32x32.
#
# Este archivo importa las clases de physical_autoencoder_red1.py, así
# que debe estar en la MISMA carpeta que ese archivo.
#
# Uso:
#   python physical_autoencoder_red2.py
# ================================================================

import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

import physical_autoencoder_red1 as P


# ================================================================
# ARGUMENTOS
# ================================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
            "--red1-checkpoint",
            default="outputs/physical_autoencoder/latent_2048/checkpoints/best.ckpt",
    )

    # Estadísticas de normalización de la Red 1. Si no se indica, se
    # busca normalization_stats.npz junto a la carpeta latent_XXXX.
    p.add_argument("--stats-path", default=None)

    p.add_argument("--data-dir", default="extracted_data")
    p.add_argument("--drive-folder-id", default=P.DRIVE_FOLDER_ID)
    p.add_argument("--hsi-file", default="simulated_hsi.npy")

    # Suavizado espectral del HSI simulado (Savitzky-Golay a lo largo de
    # las bandas) para reducir su ruido. 0 = sin suavizado. Debe ser impar.
    p.add_argument("--spectral-smooth", type=int, default=0)
    p.add_argument("--smooth-polyorder", type=int, default=3)
    p.add_argument("--output-dir", default="outputs/physical_autoencoder_red2")

    p.add_argument("--patch-size", type=int, default=32)
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

    # Pesos de las tres pérdidas (usa los mismos de la Red 1).
    p.add_argument("--weight-eps", type=float, default=1.2)
    p.add_argument("--weight-temp", type=float, default=10)
    p.add_argument("--weight-depth", type=float, default=10)
    p.add_argument("--edge-alpha", type=float, default=0)

    # Pérdida de gradiente espacial: L1 entre los gradientes de la
    # reconstrucción y los del ground truth. Penaliza textura donde el
    # ground truth es plano. 0 = desactivada.
    p.add_argument("--grad-weight", type=float, default=0)

    # Ruido gaussiano (en unidades normalizadas) que se suma al input
    # SOLO durante el entrenamiento. 0 = desactivado.
    p.add_argument("--input-noise", type=float, default=0.0)

    # Regularización de Tikhonov sobre el eje espectral de la emisividad
    # (parámetro alpha/lambda), calculada en UNIDADES FÍSICAS de
    # emisividad (se denormaliza con las estadísticas de la Red 1 antes
    # de medir la suavidad). 0 = desactivada.
    p.add_argument("--tikhonov-alpha", type=float, default=7500)

    # Si se activa, además penaliza la diferencia entre la banda 0 y la
    # última banda, como si el eje espectral fuera cíclico. Por defecto
    # NO: la banda 0 y la 255 son los extremos opuestos de un espectro
    # real, no vecinos, así que no tiene sentido forzarlos a parecerse.
    p.add_argument(
        "--tikhonov-circular",
        action="store_true",
        default=False,
    )

    p.add_argument("--visual-patch-index", type=int, default=0)

    # Reconstrucción de la escena completa.
    p.add_argument("--scene-stride", type=int, default=16)
    p.add_argument("--scene-bands", nargs="+", type=int, default=[0])

    p.add_argument("--seed", type=int, default=2026)

    return p.parse_args()


# ================================================================
# DATOS
# ================================================================

def load_hsi(path: Path, smooth_window=0, polyorder=3):
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path}.")

    hsi = np.load(path).astype(np.float32, copy=False)

    if hsi.ndim != 3 or hsi.shape[:2] != (P.H_EXPECTED, P.W_EXPECTED):
        raise ValueError(
            f"simulated_hsi incorrecto. Esperado "
            f"{(P.H_EXPECTED, P.W_EXPECTED, 'bandas')}, "
            f"recibido {hsi.shape}"
        )

    if not np.isfinite(hsi).all():
        raise ValueError("simulated_hsi contiene NaN o Inf.")

    if smooth_window > 0:
        from scipy.signal import savgol_filter

        if smooth_window % 2 == 0 or smooth_window <= polyorder:
            raise ValueError(
                "--spectral-smooth debe ser impar y mayor que "
                "--smooth-polyorder."
            )

        hsi = savgol_filter(
            hsi, smooth_window, polyorder, axis=2
        ).astype(np.float32, copy=False)

    print("\n========== SIMULATED HSI ==========")
    print("Cubo simulado:", hsi.shape)
    print(
        "Suavizado espectral:",
        f"Savitzky-Golay ventana={smooth_window}, orden={polyorder}"
        if smooth_window > 0
        else "no",
    )

    return hsi


def compute_hsi_normalization(hsi, regions):
    """Media y desviación por banda, SOLO con la región de train."""
    x0, x1 = regions["train"]
    train = hsi[:, x0:x1, :]

    mean = np.mean(train, axis=(0, 1), keepdims=True).astype(np.float32)
    std = (
        np.std(train, axis=(0, 1), keepdims=True) + 1e-6
    ).astype(np.float32)

    return {"mean": mean, "std": std}


def load_red1_stats(stats_path: Path):
    z = np.load(stats_path)

    return {
        "eps_mean": z["eps_mean"].astype(np.float32),
        "eps_std": z["eps_std"].astype(np.float32),
        "temp_mean": float(z["temp_mean"]),
        "temp_std": float(z["temp_std"]),
        "depth_mean": float(z["depth_mean"]),
        "depth_std": float(z["depth_std"]),
    }


def get_target_stats(args, physical_cube, regions):
    """
    Estadísticas de normalización de la Red 1 (las mismas, para que el
    decoder congelado produzca los mapas en la escala en que se entrenó).
    """
    if args.stats_path is not None:
        stats_path = Path(args.stats_path)
    else:
        stats_path = (
            Path(args.red1_checkpoint).parents[2]
            / "normalization_stats.npz"
        )

    recomputed = P.compute_normalization(physical_cube, regions)

    if not stats_path.exists():
        print(
            f"\nADVERTENCIA: no existe {stats_path}. "
            "Se recalculan las estadísticas con el split de train "
            "(deben coincidir con las de la Red 1)."
        )
        return recomputed

    stats = load_red1_stats(stats_path)

    same = (
        np.allclose(stats["eps_mean"], recomputed["eps_mean"], rtol=1e-4)
        and np.allclose(stats["eps_std"], recomputed["eps_std"], rtol=1e-4)
        and math.isclose(stats["temp_mean"], recomputed["temp_mean"], rel_tol=1e-4)
        and math.isclose(stats["temp_std"], recomputed["temp_std"], rel_tol=1e-4)
        and math.isclose(stats["depth_mean"], recomputed["depth_mean"], rel_tol=1e-4)
        and math.isclose(stats["depth_std"], recomputed["depth_std"], rel_tol=1e-4)
    )

    if not same:
        print(
            "\nADVERTENCIA: las estadísticas guardadas de la Red 1 NO "
            "coinciden con las recalculadas sobre estos datos. "
            "Se usan las de la Red 1."
        )

    print("\nEstadísticas de la Red 1 cargadas desde:", stats_path)

    return stats


class PairPatchDataset(Dataset):
    """
    Devuelve (parche del HSI simulado, parche del cubo físico).
    Los tensores CHW se comparten entre train/val/test (no se copian).
    """

    def __init__(self, inputs_chw, targets_chw, coordinates, patch_size=32):
        self.inputs = inputs_chw
        self.targets = targets_chw
        self.coordinates = coordinates
        self.patch_size = patch_size

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        row, col = self.coordinates[idx]
        s = self.patch_size

        return (
            self.inputs[:, row:row + s, col:col + s],
            self.targets[:, row:row + s, col:col + s],
        )


def create_loaders(inputs_chw, targets_chw, regions, args):
    datasets = {}

    for name, stride in [
        ("train", args.train_stride),
        ("val", args.val_stride),
        ("test", args.test_stride),
    ]:
        coords = P.patch_coordinates(
            P.H_EXPECTED,
            *regions[name],
            args.patch_size,
            stride,
        )
        datasets[name] = PairPatchDataset(
            inputs_chw, targets_chw, coords, args.patch_size
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
        datasets["train"],
        shuffle=True,
        generator=generator,
        **loader_args,
    )
    val_loader = DataLoader(
        datasets["val"], shuffle=False, **loader_args
    )
    test_loader = DataLoader(
        datasets["test"], shuffle=False, **loader_args
    )

    print("\n========== PATCHES ==========")
    print("Train patches:", len(datasets["train"]))
    print("Val patches  :", len(datasets["val"]))
    print("Test patches :", len(datasets["test"]))

    return (
        datasets["train"],
        datasets["val"],
        datasets["test"],
        train_loader,
        val_loader,
        test_loader,
    )


# ================================================================
# RED 2 - ENCODER MULTIMODAL SOBRE EL HSI SIMULADO
# ================================================================

class SimulatedEncoder(nn.Module):
    """
    simulated_hsi (in_bands x 32 x 32)  ──┬── enc_eps   -> 64 x 8 x 8 ─┐
      (el MISMO cubo entra a los tres)    ├── enc_temp  -> 32 x 8 x 8 ─┼─ concat
                                          └── enc_depth -> 32 x 8 x 8 ─┘  128 x 8 x 8
                                                                            ↓
                                                                   encoder de fusión
                                                                      128 x 4 x 4
                                                                            ↓
                                                                        latent_dim

    Misma estructura y mismos tamaños de canales que el PhysicalEncoder
    de la Red 1; solo cambia el número de canales de entrada.

    forward devuelve (z, skips). Los skips (16x16 y 8x8 de cada
    modalidad) tienen los mismos canales que los de la Red 1, así que
    el decoder congelado los acepta; solo se usan si ese decoder se
    entrenó con skips.
    """

    def __init__(self, latent_dim, in_bands):
        super().__init__()

        if latent_dim < 1 or latent_dim > 2048:
            raise ValueError("latent_dim debe estar entre 1 y 2048.")

        self.eps_encoder = P.ModalityEncoder(in_bands, 128, 64)
        self.temp_encoder = P.ModalityEncoder(in_bands, 16, 32)
        self.depth_encoder = P.ModalityEncoder(in_bands, 16, 32)

        self.fusion_encoder = P.EncoderBlock(128, 128)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        eps16, eps8 = self.eps_encoder(x)
        temp16, temp8 = self.temp_encoder(x)
        depth16, depth8 = self.depth_encoder(x)

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
# PÉRDIDA DE GRADIENTE
# ================================================================

def gradient_loss(pred, true):
    """
    L1 entre los gradientes espaciales de la reconstrucción y los del
    ground truth. Donde el ground truth es plano, cualquier gradiente
    de la reconstrucción (textura, ruido) se penaliza; en los bordes
    exige que sean igual de nítidos.
    """
    dx = (
        (pred[:, :, :, 1:] - pred[:, :, :, :-1])
        - (true[:, :, :, 1:] - true[:, :, :, :-1])
    )
    dy = (
        (pred[:, :, 1:, :] - pred[:, :, :-1, :])
        - (true[:, :, 1:, :] - true[:, :, :-1, :])
    )

    return dx.abs().mean() + dy.abs().mean()


def tikhonov_regularization(emissivity, eps_mean, eps_std, alpha, circular=False):
    """
    Regularización de Tikhonov sobre el eje espectral de la emisividad,
    calculada en UNIDADES FÍSICAS (no en el espacio normalizado por
    banda), para que `alpha` corresponda a una penalización real y
    comparable con la curva que se ve en la firma espectral.

    Antes esta función recibía `emissivity` ya normalizado por banda
    (misma escala con la que se entrena la Red 1: cada una de las 256
    bandas tiene su propia media/desviación). El problema es que la
    diferencia entre bandas normalizadas NO es proporcional a la
    diferencia física real: si la desviación cambia de una banda a la
    siguiente, una curva suave en unidades físicas puede verse rugosa
    en el espacio normalizado (y viceversa). Por eso subir `alpha` no
    suavizaba de forma predecible la curva real.

    emissivity: (B, 256, H, W), NORMALIZADO (salida directa del
                decoder, con la misma escala de la Red 1).
    eps_mean, eps_std: (1, 256, 1, 1) — estadísticas de la Red 1,
                usadas para denormalizar antes de medir la suavidad.
    alpha:      parámetro de regularización (lambda). Al pasar a
                unidades físicas, la magnitud de referencia cambia
                respecto a la versión anterior (normalizada): puede
                que necesites recalibrar `alpha` desde valores bajos
                (10, 100, 1000, ...) en vez de partir de 1e5.
    circular:   si True, además penaliza la diferencia entre la
                primera y la última banda, como si el eje espectral
                fuera cíclico. Por defecto False: la banda 0 y la 255
                son los extremos opuestos de un espectro real, no
                vecinos, así que no deberían forzarse a ser parecidas.
    """
    # En float32 para que el cuadrado de diferencias pequeñas no se
    # pierda con precisión mixta.
    emissivity = emissivity.float()

    # Denormalizar: "physical" queda en las mismas unidades que la
    # curva de la gráfica "Firma espectral".
    physical = emissivity * eps_std + eps_mean

    # Diferencia hacia adelante a lo largo de las bandas.
    diff = physical[:, 1:] - physical[:, :-1]

    if circular:
        # Diferencia circular entre la primera y la última banda.
        circular_diff = physical[:, 0] - physical[:, -1]
        diff = torch.cat([diff, circular_diff.unsqueeze(1)], dim=1)

    # Elevar al cuadrado y promediar.
    reg_term = torch.mean(diff ** 2)

    return alpha * reg_term


# ================================================================
# LIGHTNING MODULE
# ================================================================

class HSIAutoencoderLightning(L.LightningModule):

    def __init__(
        self,
        latent_dim,
        in_bands,
        lr=1e-3,
        weight_eps=1.0,
        weight_temp=1.0,
        weight_depth=1.0,
        edge_alpha=1.0,
        use_skips=False,
        grad_weight=0.0,
        input_noise=0.0,
        tikhonov_alpha=7000,
        tikhonov_circular=False,
        stats=None,
    ):
        super().__init__()

        # `stats` (dict con arrays de numpy) no se guarda como
        # hiperparámetro: solo se usa aquí para inicializar los
        # buffers de abajo. Los buffers sí quedan en el checkpoint,
        # así que al recargar el modelo con load_from_checkpoint no
        # hace falta volver a pasar `stats`.
        self.save_hyperparameters(ignore=["stats"])

        self.encoder = SimulatedEncoder(latent_dim, in_bands)

        # Mismo decoder de la Red 1 (con o sin skips, igual que en la
        # Red 1). Sus pesos se cargan desde el checkpoint de la Red 1
        # y queda congelado.
        self.decoder = P.PhysicalDecoder(latent_dim, use_skips)

        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)

        self.decoder.eval()

        # ------------------------------------------------------------
        # Estadísticas de emisividad de la Red 1, necesarias para
        # denormalizar eps_pred antes de medir su suavidad espectral
        # (ver tikhonov_regularization). Se guardan como buffers (no
        # entrenables): quedan en el state_dict del checkpoint y se
        # restauran solos al cargarlo con load_from_checkpoint.
        # ------------------------------------------------------------
        if stats is not None:
            eps_mean = torch.as_tensor(
                stats["eps_mean"], dtype=torch.float32
            ).reshape(1, P.N_EMISSIVITY, 1, 1)
            eps_std = torch.as_tensor(
                stats["eps_std"], dtype=torch.float32
            ).reshape(1, P.N_EMISSIVITY, 1, 1)
        else:
            # Se sobreescriben al cargar un checkpoint que ya los
            # tenga guardados (load_state_dict), así que estos valores
            # por defecto solo importan si se instancia el módulo sin
            # stats Y sin cargar checkpoint (no debería pasar en uso
            # normal de este script).
            eps_mean = torch.zeros(1, P.N_EMISSIVITY, 1, 1)
            eps_std = torch.ones(1, P.N_EMISSIVITY, 1, 1)

        self.register_buffer("eps_mean_buf", eps_mean)
        self.register_buffer("eps_std_buf", eps_std)

    def train(self, mode=True):
        # El decoder SIEMPRE en eval: BatchNorm no debe actualizar sus
        # estadísticas mientras entrenamos los encoders.
        super().train(mode)
        self.decoder.eval()
        return self

    def forward(self, x):
        z, skips = self.encoder(x)
        reconstruction = self.decoder(z, skips)
        return reconstruction, z

    def calculate_losses(self, batch, return_reconstruction=False):
        inputs, targets = batch

        reconstruction, _ = self(inputs)

        eps_true = targets[:, :256]
        temp_true = targets[:, 256:257]
        depth_true = targets[:, 257:258]

        eps_pred = reconstruction[:, :256]
        temp_pred = reconstruction[:, 256:257]
        depth_pred = reconstruction[:, 257:258]

        alpha = self.hparams.edge_alpha

        loss_eps = P.pixel_weighted_mse(eps_pred, eps_true, alpha)
        loss_temp = P.pixel_weighted_mse(temp_pred, temp_true, alpha)
        loss_depth = P.pixel_weighted_mse(depth_pred, depth_true, alpha)

        grad_weight = self.hparams.grad_weight

        if grad_weight > 0:
            loss_eps = loss_eps + grad_weight * gradient_loss(
                eps_pred, eps_true
            )
            loss_temp = loss_temp + grad_weight * gradient_loss(
                temp_pred, temp_true
            )
            loss_depth = loss_depth + grad_weight * gradient_loss(
                depth_pred, depth_true
            )

        # Regularización de Tikhonov: suavidad espectral de la
        # emisividad reconstruida (solo emisividad).
        tikhonov_alpha = self.hparams.tikhonov_alpha

        if tikhonov_alpha > 0:
            loss_eps = loss_eps + tikhonov_regularization(
                eps_pred,
                self.eps_mean_buf,
                self.eps_std_buf,
                tikhonov_alpha,
                circular=False,
            )

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

        return loss_total, loss_eps, loss_temp, loss_depth

    def training_step(self, batch, batch_idx):
        noise = self.hparams.input_noise

        if noise > 0:
            inputs, targets = batch
            batch = (
                inputs + noise * torch.randn_like(inputs),
                targets,
            )

        total, eps, temp, depth = self.calculate_losses(batch)

        self.log("train_total", total, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_eps", eps, on_step=False, on_epoch=True)
        self.log("train_temp", temp, on_step=False, on_epoch=True)
        self.log("train_depth", depth, on_step=False, on_epoch=True)

        return total

    def validation_step(self, batch, batch_idx):
        _, targets = batch

        (
            total,
            eps,
            temp,
            depth,
            reconstruction,
        ) = self.calculate_losses(batch, return_reconstruction=True)

        self.log("val_total", total, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_eps", eps, on_step=False, on_epoch=True)
        self.log("val_temp", temp, on_step=False, on_epoch=True)
        self.log("val_depth", depth, on_step=False, on_epoch=True)

        # Métricas planas (sin pesos) para early stopping / checkpoint.
        # Se promedian por igual las 3 modalidades.
        mae_eps = F.l1_loss(reconstruction[:, :256], targets[:, :256])
        mae_temp = F.l1_loss(reconstruction[:, 256:257], targets[:, 256:257])
        mae_depth = F.l1_loss(reconstruction[:, 257:258], targets[:, 257:258])
        val_mae = (mae_eps + mae_temp + mae_depth) / 3.0

        mse_eps = F.mse_loss(reconstruction[:, :256], targets[:, :256])
        mse_temp = F.mse_loss(reconstruction[:, 256:257], targets[:, 256:257])
        mse_depth = F.mse_loss(reconstruction[:, 257:258], targets[:, 257:258])
        val_mse = (mse_eps + mse_temp + mse_depth) / 3.0

        self.log("val_mae", val_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mse", val_mse, on_step=False, on_epoch=True)

        return total

    def test_step(self, batch, batch_idx):
        total, eps, temp, depth = self.calculate_losses(batch)

        self.log("test_total", total, on_step=False, on_epoch=True)
        self.log("test_eps", eps, on_step=False, on_epoch=True)
        self.log("test_temp", temp, on_step=False, on_epoch=True)
        self.log("test_depth", depth, on_step=False, on_epoch=True)

        return total

    def configure_optimizers(self):
        # Solo los encoders: el decoder está congelado.
        return torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.hparams.lr,
        )


# ================================================================
# MÉTRICAS EN UNIDADES FÍSICAS
# ================================================================

@torch.no_grad()
def physical_metrics_red2(model, loader, device, stats):
    """
    Mismas métricas que la Red 1. El error en unidades físicas es
    (pred - true) * std, porque la media se cancela.
    """
    model.eval()
    model.to(device)

    eps_std = torch.tensor(
        stats["eps_std"], dtype=torch.float32, device=device
    ).reshape(1, 256, 1, 1)

    temp_std = float(stats["temp_std"])
    depth_std = float(stats["depth_std"])

    accum = {
        "eps": [0.0, 0.0, 0],
        "temp": [0.0, 0.0, 0],
        "depth": [0.0, 0.0, 0],
    }

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        prediction, _ = model(inputs)

        errors = {
            "eps": (prediction[:, :256] - targets[:, :256]) * eps_std,
            "temp": (prediction[:, 256:257] - targets[:, 256:257]) * temp_std,
            "depth": (prediction[:, 257:258] - targets[:, 257:258]) * depth_std,
        }

        for key, err in errors.items():
            accum[key][0] += torch.sum(err ** 2).item()
            accum[key][1] += torch.sum(torch.abs(err)).item()
            accum[key][2] += err.numel()

    metrics = {}

    for key, (sq, ab, n) in accum.items():
        metrics[f"{key}_rmse"] = math.sqrt(sq / n)
        metrics[f"{key}_mae"] = ab / n

    return metrics


# ================================================================
# VISUALIZACIÓN (reutiliza las funciones de la Red 1)
# ================================================================

class _VisualDataset(Dataset):
    """Devuelve [target(258) | hsi(256)] concatenados en canales."""

    def __init__(self, pair_dataset):
        self.pair_dataset = pair_dataset

    def __len__(self):
        return len(self.pair_dataset)

    def __getitem__(self, idx):
        inputs, targets = self.pair_dataset[idx]
        return torch.cat([targets, inputs], dim=0)


class _VisualAdapter(nn.Module):
    """
    Adaptador para reutilizar save_visual_validation y
    save_full_scene_reconstruction de la Red 1. Recibe
    [target(258) | hsi(256)], usa solo el HSI y devuelve
    (reconstrucción, z).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        hsi = x[:, P.N_PHYSICAL:]
        return self.model(hsi)


# ================================================================
# MAIN
# ================================================================

def main():

    args = parse_args()

    L.seed_everything(args.seed, workers=True)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1. Decoder de la Red 1
    # ------------------------------------------------------------

    ckpt_path = Path(args.red1_checkpoint)

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No se encontró el checkpoint de la Red 1: {ckpt_path}"
        )

    red1 = P.PhysicalAutoencoderLightning.load_from_checkpoint(
        str(ckpt_path),
        map_location="cpu",
    )

    # Si la Red 1 se entrenó con skips, el decoder los espera y los
    # encoders de la Red 2 los producen desde el HSI simulado.
    use_skips = bool(red1.decoder.use_skips)

    latent_dim = int(red1.hparams["latent_dim"])
    decoder_state = red1.decoder.state_dict()
    del red1

    print("\n========== RED 1 ==========")
    print("Checkpoint :", ckpt_path)
    print("latent_dim :", latent_dim)
    print(
        "Skips      :",
        "sí (los produce el encoder de la Red 2 desde el HSI)"
        if use_skips
        else "no (el decoder recibe solo z)",
    )

    # ------------------------------------------------------------
    # 2. Datos
    # ------------------------------------------------------------

    data_dir = P.ensure_extracted_data(
        Path(args.data_dir),
        args.drive_folder_id,
        required=P.REQUIRED_DATA_FILES + [args.hsi_file],
    )

    physical_cube = P.load_data(data_dir)
    hsi = load_hsi(
        data_dir / args.hsi_file,
        args.spectral_smooth,
        args.smooth_polyorder,
    )
    in_bands = hsi.shape[2]

    regions = P.spatial_regions(
        P.W_EXPECTED,
        args.train_fraction,
        args.val_fraction,
    )

    print("\n========== REGIONES ==========")

    for name, (x0, x1) in regions.items():
        print(f"{name:5s}: {x0} -> {x1} (ancho={x1 - x0})")

    # Targets: normalizados con las estadísticas de la Red 1.
    stats = get_target_stats(args, physical_cube, regions)

    targets_norm = P.normalize_cube(physical_cube, stats)
    targets_chw = (
        torch.from_numpy(targets_norm).permute(2, 0, 1).contiguous()
    )
    del targets_norm

    # Inputs: HSI normalizado por banda con estadísticas de TRAIN.
    hsi_stats = compute_hsi_normalization(hsi, regions)

    inputs_norm = (
        (hsi - hsi_stats["mean"]) / hsi_stats["std"]
    ).astype(np.float32, copy=False)
    del hsi

    inputs_chw = (
        torch.from_numpy(inputs_norm).permute(2, 0, 1).contiguous()
    )
    del inputs_norm

    np.savez(
        output_dir / "hsi_normalization_stats.npz",
        mean=hsi_stats["mean"],
        std=hsi_stats["std"],
    )

    (
        train_ds,
        val_ds,
        test_ds,
        train_loader,
        val_loader,
        test_loader,
    ) = create_loaders(inputs_chw, targets_chw, regions, args)

    inputs_batch, targets_batch = next(iter(train_loader))

    print("\n========== VERIFICACIÓN ==========")
    print("Batch input :", tuple(inputs_batch.shape))
    print("Batch target:", tuple(targets_batch.shape))
    print("Métrica early stopping / checkpoint:", args.monitor)

    # ------------------------------------------------------------
    # 3. Modelo (decoder congelado con los pesos de la Red 1)
    # ------------------------------------------------------------

    model = HSIAutoencoderLightning(
        latent_dim=latent_dim,
        in_bands=in_bands,
        lr=args.lr,
        weight_eps=args.weight_eps,
        weight_temp=args.weight_temp,
        weight_depth=args.weight_depth,
        edge_alpha=args.edge_alpha,
        use_skips=use_skips,
        grad_weight=args.grad_weight,
        input_noise=args.input_noise,
        tikhonov_alpha=args.tikhonov_alpha,
        tikhonov_circular=args.tikhonov_circular,
        stats=stats,
    )

    model.decoder.load_state_dict(decoder_state)
    del decoder_state

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print("\n========== PARÁMETROS ==========")
    print(f"Entrenables (encoders): {n_train:,}")
    print(f"Congelados  (decoder) : {n_frozen:,}")

    # ------------------------------------------------------------
    # 4. Entrenamiento
    # ------------------------------------------------------------

    # Cada corrida sobreescribe la anterior: se borran los checkpoints
    # y logs viejos, así best.ckpt siempre es el de esta corrida
    # (Lightning, si no, guardaría best-v1.ckpt, best-v2.ckpt...).
    shutil.rmtree(output_dir / "checkpoints", ignore_errors=True)
    shutil.rmtree(output_dir / "logs", ignore_errors=True)

    checkpoint = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
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
        save_dir=str(output_dir),
        name="logs",
    )

    precision = "16-mixed" if torch.cuda.is_available() else "32-true"

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        precision=precision,
        callbacks=[checkpoint, early_stopping],
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
        raise RuntimeError("No se generó checkpoint.")

    best_model = HSIAutoencoderLightning.load_from_checkpoint(best_path)

    val_result = trainer.validate(
        best_model, dataloaders=val_loader, verbose=False
    )[0]

    test_result = trainer.test(
        best_model, dataloaders=test_loader, verbose=False
    )[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    physical = physical_metrics_red2(
        best_model, test_loader, device, stats
    )

    # ------------------------------------------------------------
    # 5. Visualizaciones
    # ------------------------------------------------------------

    visual_dir = output_dir / "visual_validation"

    adapter = _VisualAdapter(best_model)

    P.save_visual_validation(
        adapter,
        _VisualDataset(val_ds),
        args.visual_patch_index,
        device,
        stats,
        visual_dir,
        latent_dim,
    )

    # Cubo [target(258) | hsi(256)] para reutilizar la función de la escena.
    scene_cube = np.concatenate(
        [
            targets_chw.permute(1, 2, 0).numpy(),
            inputs_chw.permute(1, 2, 0).numpy(),
        ],
        axis=2,
    )

    P.save_full_scene_reconstruction(
        adapter,
        scene_cube,
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

    del scene_cube

    # ------------------------------------------------------------
    # 6. Resultados y artefacto
    # ------------------------------------------------------------

    row = {
        "latent_dim": latent_dim,
        "val_total": float(val_result["val_total"]),
        "val_eps": float(val_result["val_eps"]),
        "val_temp": float(val_result["val_temp"]),
        "val_depth": float(val_result["val_depth"]),
        "val_mae": float(val_result["val_mae"]),
        "val_mse": float(val_result["val_mse"]),
        "test_total": float(test_result["test_total"]),
        "test_eps": float(test_result["test_eps"]),
        "test_temp": float(test_result["test_temp"]),
        "test_depth": float(test_result["test_depth"]),
        **physical,
        "checkpoint": best_path,
    }

    pd.DataFrame([row]).to_csv(output_dir / "summary.csv", index=False)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2, ensure_ascii=False)

    artifact = {
        "latent_dim": latent_dim,
        "in_bands": in_bands,
        "skip_connections": use_skips,
        "spectral_smooth": args.spectral_smooth,
        "smooth_polyorder": args.smooth_polyorder,
        "encoder_state_dict": best_model.encoder.state_dict(),
        "decoder_state_dict": best_model.decoder.state_dict(),
        "model_state_dict": best_model.state_dict(),
        "hsi_normalization": hsi_stats,
        "normalization": stats,
        "input_shape": (in_bands, args.patch_size, args.patch_size),
        "output_shape": (P.N_PHYSICAL, args.patch_size, args.patch_size),
        "image_shape": (P.H_EXPECTED, P.W_EXPECTED),
        "red1_checkpoint": str(ckpt_path),
    }

    torch.save(artifact, output_dir / f"physical_autoencoder_red2_z{latent_dim}.pt")

    print("\n")
    print("=" * 70)
    print("ENTRENAMIENTO RED 2 TERMINADO")
    print("=" * 70)

    for key, value in row.items():
        if key != "checkpoint":
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()