"""Funcions senzilles d'entrada/sortida."""

import json
from pathlib import Path
from typing import Any


def ensure_directories(*paths: Path) -> None:
    """Crea els directoris si no existeixen."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_json(data: dict[str, Any], path: Path) -> None:
    """Guarda un diccionari com a JSON llegible."""
    ensure_directories(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_json(path: Path) -> dict[str, Any]:
    """Carrega un JSON des de disc."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
