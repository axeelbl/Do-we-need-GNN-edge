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
    k_neighbors: int = K_NEIGHBORS,
) -> Data:
    """Combina train/test y crea un objeto Data de PyTorch Geometric."""

    # Comprobamos que features y labels tengan formas compatibles.
    _validate_inputs(train_features, train_labels, test_features, test_labels)

    # Unimos train y test en un unico conjunto de nodos.
    x = torch.cat([train_features, test_features], dim=0).float()
    y = torch.cat([train_labels, test_labels], dim=0).long()

    # Guardamos cuantos nodos pertenecen a train y cuantos a test.
    num_train = train_features.shape[0]
    num_test = test_features.shape[0]
    num_nodes = x.shape[0]

    # Creamos las aristas conectando cada nodo con sus vecinos mas parecidos.
    edge_index = _build_knn_edge_index(x, k_neighbors)

    # Las mascaras indican que nodos se usan para entrenar y evaluar.
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:num_train] = True
    test_mask[num_train : num_train + num_test] = True

    # Data es el formato estandar de PyTorch Geometric.
    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        test_mask=test_mask,
    )


def _build_knn_edge_index(features: torch.Tensor, k_neighbors: int) -> torch.Tensor:
    """Construye aristas bidireccionales con KNN evitando duplicados."""

    # K debe ser positivo para que exista al menos un vecino.
    if k_neighbors <= 0:
        raise ValueError("k_neighbors debe ser mayor que 0.")

    # Con un solo nodo no se pueden crear relaciones.
    num_nodes = features.shape[0]
    if num_nodes < 2:
        raise ValueError("Se necesitan al menos 2 nodos para construir un grafo.")

    # Si el dataset es pequeno, K no puede superar el numero de otros nodos.
    effective_k = min(k_neighbors, num_nodes - 1)

    # sklearn trabaja con arrays de numpy en CPU.
    features_np = features.detach().cpu().numpy().astype(np.float32)

    # Pedimos K+1 vecinos porque el primer vecino suele ser el propio nodo.
    knn = NearestNeighbors(n_neighbors=effective_k + 1, metric="euclidean")
    knn.fit(features_np)
    neighbors = knn.kneighbors(features_np, return_distance=False)

    # Usamos un set para evitar aristas duplicadas.
    edges: set[tuple[int, int]] = set()
    for source, neighbor_indices in enumerate(neighbors):
        # Saltamos neighbor_indices[0], que normalmente es el propio nodo.
        for target in neighbor_indices[1:]:
            target_index = int(target)

            # Anadir las dos direcciones facilita el uso con GCNConv.
            edges.add((source, target_index))
            edges.add((target_index, source))

    # edge_index debe tener forma (2, num_edges).
    ordered_edges = sorted(edges)
    return torch.tensor(ordered_edges, dtype=torch.long).t().contiguous()


def _validate_inputs(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
) -> None:
    """Comprueba formas basicas antes de construir el grafo."""

    # Las features deben ser matrices: una fila por nodo.
    if train_features.ndim != 2 or test_features.ndim != 2:
        raise ValueError("Las features deben tener forma (num_samples, feature_dim).")

    # Train y test deben usar la misma representacion de features.
    if train_features.shape[1] != test_features.shape[1]:
        raise ValueError("train_features y test_features deben tener la misma dimension.")

    # Cada muestra de train necesita una etiqueta.
    if train_features.shape[0] != train_labels.shape[0]:
        raise ValueError("train_features y train_labels tienen tamanos distintos.")

    # Cada muestra de test necesita una etiqueta.
    if test_features.shape[0] != test_labels.shape[0]:
        raise ValueError("test_features y test_labels tienen tamanos distintos.")

    # Las etiquetas se esperan como vector 1D.
    if train_labels.ndim != 1 or test_labels.ndim != 1:
        raise ValueError("Las etiquetas deben tener forma (num_samples,).")
