"""Modelo Full-GNN para clasificacion de nodos."""

import torch
from torch import nn
import torch.nn.functional as functional
from torch_geometric.nn import GCNConv

from config import DROPOUT, FULL_GNN_HIDDEN_DIM, INPUT_DIM, NUM_CLASSES


class FullGNN(nn.Module):
    """GCN de tres capas con mayor capacidad que Tiny-GNN."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = FULL_GNN_HIDDEN_DIM,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()

        # Dropout reduce sobreajuste apagando parte de las activaciones.
        self.dropout = dropout

        # Primera capa: pasa de features de imagen a representacion oculta.
        self.conv1 = GCNConv(input_dim, hidden_dim)

        # Segunda capa: procesa la informacion ya mezclada por el grafo.
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Tercera capa: produce un logit por clase para cada nodo.
        self.conv3 = GCNConv(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Devuelve logits por nodo."""

        # Capa 1: agregacion de vecinos + activacion no lineal.
        x = self.conv1(x, edge_index)
        x = functional.relu(x)
        x = functional.dropout(x, p=self.dropout, training=self.training)

        # Capa 2: nueva agregacion sobre las representaciones ocultas.
        x = self.conv2(x, edge_index)
        x = functional.relu(x)
        x = functional.dropout(x, p=self.dropout, training=self.training)

        # Capa final: salida de clasificacion sin softmax.
        # CrossEntropyLoss espera logits directamente.
        return self.conv3(x, edge_index)
