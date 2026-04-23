"""Rutinas minimas de entrenamiento."""

import torch
from torch import nn
from torch.optim import Optimizer
from torch_geometric.data import Data


def train_one_epoch(
    model: nn.Module,
    data: Data,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: torch.device | str,
) -> float:
    """Entrena un modelo durante una epoca y devuelve la perdida."""

    # Nos aseguramos de que modelo y datos esten en el mismo dispositivo.
    model.to(device)
    data = data.to(device)

    # Modo entrenamiento: activa dropout y permite calcular gradientes.
    model.train()

    # Limpiamos gradientes anteriores antes de calcular los nuevos.
    optimizer.zero_grad()

    # Forward completo sobre todos los nodos del grafo.
    logits = model(data.x, data.edge_index)

    # La perdida se calcula solo con los nodos de entrenamiento.
    loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])

    # Backpropagation: calcula gradientes.
    loss.backward()

    # Actualiza los parametros del modelo.
    optimizer.step()

    # Devolvemos un float normal para guardarlo facilmente en JSON.
    return float(loss.item())
