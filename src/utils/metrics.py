"""Helpers per a mètriques i exportació de resultats."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from torch import nn

from config import AppConfig


def build_initial_metrics(config: AppConfig) -> dict[str, Any]:
    """Crea la metadata inicial del fitxer de mètriques."""

    return {
        "project": "tfg_tinygnn_physics",
        "status": "initialized",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "simulation": {
            "grid_size": config.grid.grid_size,
            "num_timesteps": config.grid.num_timesteps,
            "train_steps": config.grid.train_steps,
            "val_steps": config.grid.val_steps,
            "test_steps": config.grid.test_steps,
            "alpha": config.grid.alpha,
            "dt": config.grid.dt,
            "dx": config.grid.dx,
        },
        "training": {
            "epochs": config.training.epochs,
            "learning_rate": config.training.learning_rate,
            "physics_lambda": config.training.physics_lambda,
            "device": config.training.device,
            "status": "pending",
        },
        "models": {
            "mlp_baseline": {"uses_graph": False},
            "full_gnn": {"uses_graph": True},
            "tiny_gnn": {"uses_graph": True},
            "tiny_gnn_pinn": {"uses_graph": True, "physics_informed": True},
        },
        "runs": [],
    }


def count_trainable_parameters(model: nn.Module) -> int:
    """Compta els paràmetres entrenables d'un model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_metrics_dataframe(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Converteix mètriques de models en un DataFrame."""

    rows = [{"model": model_name, **metrics} for model_name, metrics in results.items()]
    return pd.DataFrame(rows)


def save_metrics(path: Path, metrics: dict[str, Any]) -> None:
    """Guarda mètriques en JSON llegible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
        f.write("\n")


def save_metrics_dataframe(path: Path, df: pd.DataFrame) -> None:
    """Guarda DataFrame de mètriques en CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
