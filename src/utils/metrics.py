"""Helpers para crear y preparar metricas del experimento."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from torch import nn

from config import AppConfig


def build_initial_metrics(config: AppConfig, class_names: Iterable[str]) -> dict[str, Any]:
    """Crea el fichero de metricas inicial de la fase base."""

    # Este diccionario describe el estado inicial del proyecto.
    return {
        "project": "tfg_gnn",
        "status": "initialized",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": config.data.dataset_name,
            "raw_dir": str(config.data.raw_dir),
            "classes": list(class_names),
        },
        "graph": {
            "k_neighbors": config.graph.k_neighbors,
            "feature_dim": config.graph.feature_dim,
            "status": "pending",
        },
        "training": {
            "epochs": config.training.epochs,
            "learning_rate": config.training.learning_rate,
            "device": config.training.device,
            "status": "pending",
        },
        "models": {
            "full_gnn": {"status": "pending"},
            "tiny_gnn": {"status": "pending"},
        },
        "runs": [],
    }


def count_trainable_parameters(model: nn.Module) -> int:
    """Cuenta los parametros entrenables de un modelo."""

    # Solo se cuentan parametros que se actualizan durante entrenamiento.
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def save_metrics(path: Path, metrics: dict[str, Any]) -> None:
    """Guarda metricas en formato JSON legible."""

    # Creamos la carpeta de resultados si todavia no existe.
    path.parent.mkdir(parents=True, exist_ok=True)

    # indent=2 hace que el JSON sea facil de leer.
    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
        file.write("\n")
