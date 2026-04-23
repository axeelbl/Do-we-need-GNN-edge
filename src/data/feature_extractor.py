"""Extraccion simple de features para imagenes."""

import torch
import torch.nn.functional as functional

from config import NORMALIZE_FEATURES


def extract_features(
    images: torch.Tensor,
    normalize: bool = NORMALIZE_FEATURES,
) -> torch.Tensor:
    """Convierte imagenes en vectores mediante flatten y normalizacion L2."""

    # Esperamos un batch de imagenes: (N, C, H, W).
    if images.ndim != 4:
        raise ValueError("images debe tener forma (num_images, channels, height, width).")

    # No tiene sentido extraer features de un conjunto vacio.
    if images.shape[0] == 0:
        raise ValueError("images no puede estar vacio.")

    # Flatten: cada imagen pasa de (3, 32, 32) a un vector de 3072 valores.
    features = images.float().reshape(images.shape[0], -1)

    # Normalizacion L2: deja todos los vectores en una escala comparable.
    if normalize:
        features = functional.normalize(features, p=2, dim=1)

    # Devuelve una matriz con forma (num_imagenes, num_features).
    return features
