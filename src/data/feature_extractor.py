"""Extraccion simple de features para imagenes."""

import torch
import torch.nn.functional as functional

from config import IMAGES_PER_NODE, NORMALIZE_FEATURES


def extract_features(
    images: torch.Tensor,
    normalize: bool = NORMALIZE_FEATURES,
) -> torch.Tensor:
    """Convierte imagenes en vectores mediante flatten y normalizacion L2."""

    if images.ndim != 4:
        raise ValueError("images debe tener forma (num_images, channels, height, width).")

    if images.shape[0] == 0:
        raise ValueError("images no puede estar vacio.")

    features = images.float().reshape(images.shape[0], -1)

    if normalize:
        features = functional.normalize(features, p=2, dim=1)

    return features


def build_node_features_from_images(
    images: torch.Tensor,
    labels: torch.Tensor,
    images_per_node: int = IMAGES_PER_NODE,
    normalize: bool = NORMALIZE_FEATURES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Agrupa varias imagenes de la misma clase para crear nodos."""

    _validate_grouping_inputs(images, labels, images_per_node)

    image_features = extract_features(images, normalize=False)

    node_features: list[torch.Tensor] = []
    node_labels: list[int] = []
    node_image_counts: list[int] = []

    for class_id in sorted(torch.unique(labels).tolist()):
        class_mask = labels == int(class_id)
        class_indices = torch.where(class_mask)[0]

        for start in range(0, class_indices.numel(), images_per_node):
            group_indices = class_indices[start : start + images_per_node]

            group_features = image_features[group_indices]
            node_features.append(group_features.mean(dim=0))
            node_labels.append(int(class_id))
            node_image_counts.append(int(group_indices.numel()))

    features = torch.stack(node_features).float()
    if normalize:
        features = functional.normalize(features, p=2, dim=1)

    return (
        features,
        torch.tensor(node_labels, dtype=torch.long),
        torch.tensor(node_image_counts, dtype=torch.long),
    )


def _validate_grouping_inputs(
    images: torch.Tensor,
    labels: torch.Tensor,
    images_per_node: int,
) -> None:
    """Comprueba que las imagenes se pueden agrupar en nodos."""

    if images_per_node <= 0:
        raise ValueError("images_per_node debe ser mayor que 0.")
    if labels.ndim != 1:
        raise ValueError("labels debe tener forma (num_images,).")
    if images.shape[0] != labels.shape[0]:
        raise ValueError("images y labels deben tener el mismo numero de muestras.")
