"""Funciones minimas de evaluacion."""

import time

import torch
from torch import nn
from torch_geometric.data import Data

from training.trainer import model_forward


def evaluate_accuracy(model: nn.Module, data: Data, mask: torch.Tensor) -> float:
    """Calcula accuracy sobre los nodos indicados por una mascara."""

    model.eval()

    with torch.no_grad():
        logits = model_forward(model, data)
        predictions = logits.argmax(dim=1)
        selected = mask.bool()
        total = int(selected.sum().item())
        if total == 0:
            return 0.0

        correct = int((predictions[selected] == data.y[selected]).sum().item())
        return correct / total


def measure_inference_time(model: nn.Module, data: Data) -> float:
    """Mide el tiempo de un forward completo en segundos."""

    model.eval()

    with torch.no_grad():
        start_time = time.perf_counter()
        _ = model_forward(model, data)
        end_time = time.perf_counter()

    return end_time - start_time
