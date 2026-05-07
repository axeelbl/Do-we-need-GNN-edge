"""Construccion de un grafo de similitud con k-nearest neighbors."""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

from config import K_NEIGHBORS


def build_similarity_graph(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    train_node_image_counts: torch.Tensor | None = None,
    test_node_image_counts: torch.Tensor | None = None,
    k_neighbors: int = K_NEIGHBORS,
) -> Data:
    """Combina train/test y crea un objeto Data de PyTorch Geometric."""

    _validate_inputs(train_features, train_labels, test_features, test_labels)

    x = torch.cat([train_features, test_features], dim=0).float()
    y = torch.cat([train_labels, test_labels], dim=0).long()

    node_image_counts = _build_node_image_counts(
        train_features=train_features,
        test_features=test_features,
        train_node_image_counts=train_node_image_counts,
        test_node_image_counts=test_node_image_counts,
    )

    num_train = train_features.shape[0]
    num_test = test_features.shape[0]
    num_nodes = x.shape[0]

    edge_index = _build_knn_edge_index(x, k_neighbors)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:num_train] = True
    test_mask[num_train : num_train + num_test] = True

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        test_mask=test_mask,
        node_image_counts=node_image_counts,
    )


def _build_node_image_counts(
    train_features: torch.Tensor,
    test_features: torch.Tensor,
    train_node_image_counts: torch.Tensor | None,
    test_node_image_counts: torch.Tensor | None,
) -> torch.Tensor:
    """Crea el vector con el numero de imagenes que representa cada nodo."""

    if train_node_image_counts is None:
        train_node_image_counts = torch.ones(train_features.shape[0], dtype=torch.long)
    if test_node_image_counts is None:
        test_node_image_counts = torch.ones(test_features.shape[0], dtype=torch.long)

    if train_node_image_counts.shape[0] != train_features.shape[0]:
        raise ValueError("train_node_image_counts no coincide con train_features.")
    if test_node_image_counts.shape[0] != test_features.shape[0]:
        raise ValueError("test_node_image_counts no coincide con test_features.")

    return torch.cat([train_node_image_counts, test_node_image_counts], dim=0).long()


def _build_knn_edge_index(features: torch.Tensor, k_neighbors: int) -> torch.Tensor:
    """Construye aristas bidireccionales con KNN evitando duplicados."""

    if k_neighbors <= 0:
        raise ValueError("k_neighbors debe ser mayor que 0.")

    num_nodes = features.shape[0]
    if num_nodes < 2:
        raise ValueError("Se necesitan al menos 2 nodos para construir un grafo.")

    effective_k = min(k_neighbors, num_nodes - 1)

    features_np = features.detach().cpu().numpy().astype(np.float32)

    knn = NearestNeighbors(n_neighbors=effective_k + 1, metric="euclidean")
    knn.fit(features_np)
    neighbors = knn.kneighbors(features_np, return_distance=False)

    edges: set[tuple[int, int]] = set()
    for source, neighbor_indices in enumerate(neighbors):
        for target in neighbor_indices[1:]:
            target_index = int(target)

            edges.add((source, target_index))
            edges.add((target_index, source))

    ordered_edges = sorted(edges)
    return torch.tensor(ordered_edges, dtype=torch.long).t().contiguous()


def _validate_inputs(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
) -> None:
    """Comprueba formas basicas antes de construir el grafo."""

    if train_features.ndim != 2 or test_features.ndim != 2:
        raise ValueError("Las features deben tener forma (num_samples, feature_dim).")

    if train_features.shape[1] != test_features.shape[1]:
        raise ValueError("train_features y test_features deben tener la misma dimension.")

    if train_features.shape[0] != train_labels.shape[0]:
        raise ValueError("train_features y train_labels tienen tamanos distintos.")

    if test_features.shape[0] != test_labels.shape[0]:
        raise ValueError("test_features y test_labels tienen tamanos distintos.")

    if train_labels.ndim != 1 or test_labels.ndim != 1:
        raise ValueError("Las etiquetas deben tener forma (num_samples,).")
