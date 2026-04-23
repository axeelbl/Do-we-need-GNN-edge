"""Modelo Tiny-GNN para clasificacion de nodos."""

import torch
from torch import nn
import torch.nn.functional as functional
from torch_geometric.nn import GCNConv

from config import INPUT_DIM, NUM_CLASSES, TINY_GNN_HIDDEN_DIM


class TinyGNN(nn.Module):
    """GCN reducida de dos capas para comparar coste y rendimiento."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = TINY_GNN_HIDDEN_DIM,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()

        # Primera capa pequeña: reduce las features a una representacion compacta.
        self.conv1 = GCNConv(input_dim, hidden_dim)

        # Capa final: produce los logits de las 10 clases.
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Devuelve logits por nodo."""

        # Una sola capa oculta para mantener el modelo ligero.
        x = self.conv1(x, edge_index)
        x = functional.relu(x)

        # Salida directa a clases. No aplicamos softmax aqui.
        return self.conv2(x, edge_index)
