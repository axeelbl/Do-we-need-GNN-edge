"""Funciones minimas de evaluacion."""

import time

import torch
from torch import nn
from torch_geometric.data import Data


def evaluate_accuracy(model: nn.Module, data: Data, mask: torch.Tensor) -> float:
    """Calcula accuracy sobre los nodos indicados por una mascara."""

    # Modo evaluacion: desactiva dropout y comportamiento de entrenamiento.
    model.eval()

    # En evaluacion no necesitamos guardar gradientes.
    with torch.no_grad():
        # Calculamos logits para todos los nodos.
        logits = model(data.x, data.edge_index)

        # La clase predicha es el indice con logit mas alto.
        predictions = logits.argmax(dim=1)

        # La mascara decide si medimos train o test.
        selected = mask.bool()
        total = int(selected.sum().item())
        if total == 0:
            return 0.0

        # Comparamos predicciones y etiquetas reales solo en los nodos seleccionados.
        correct = int((predictions[selected] == data.y[selected]).sum().item())
        return correct / total


def measure_inference_time(model: nn.Module, data: Data) -> float:
    """Mide el tiempo de un forward completo en segundos."""

    # Medimos inferencia en modo evaluacion.
    model.eval()

    with torch.no_grad():
        # perf_counter da una medida precisa de tiempo transcurrido.
        start_time = time.perf_counter()
        _ = model(data.x, data.edge_index)
        end_time = time.perf_counter()

    # Tiempo total de un forward completo.
    return end_time - start_time
