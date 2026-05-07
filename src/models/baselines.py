"""Modelos baseline para comparar contra las GNN."""

import torch
from torch import nn
import torch.nn.functional as functional

from config import DROPOUT, INPUT_DIM, NUM_CLASSES, TINY_GNN_HIDDEN_DIM


class RandomBaseline(nn.Module):
    """Baseline que predice clases aleatorias."""

    uses_graph = False

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve logits aleatorios para cada nodo."""

        return torch.randn(x.shape[0], self.num_classes, device=x.device)


class SimpleMLPBaseline(nn.Module):
    """Baseline sin grafo que clasifica usando solo features de nodos."""

    uses_graph = False

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = TINY_GNN_HIDDEN_DIM,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve logits por nodo ignorando el grafo."""

        x = self.fc1(x)
        x = functional.relu(x)
        x = functional.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)
