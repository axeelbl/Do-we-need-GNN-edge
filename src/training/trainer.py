"""Rutinas minimas de entrenamiento."""

import torch
from torch import nn
from torch.optim import Optimizer
from torch_geometric.data import Data


def model_forward(model: nn.Module, data: Data) -> torch.Tensor:
    """Ejecuta el forward segun si el modelo usa grafo o no."""

    if getattr(model, "uses_graph", True):
        return model(data.x, data.edge_index)
    return model(data.x)


def train_one_epoch(
    model: nn.Module,
    data: Data,
    optimizer: Optimizer | None,
    loss_fn: nn.Module,
    device: torch.device | str,
) -> float:
    """Entrena un modelo durante una epoca y devuelve la perdida."""

    model.to(device)
    data = data.to(device)

    if optimizer is None:
        model.eval()
        with torch.no_grad():
            logits = model_forward(model, data)
            loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
        return float(loss.item())

    model.train()
    optimizer.zero_grad()

    logits = model_forward(model, data)
    loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    return float(loss.item())
