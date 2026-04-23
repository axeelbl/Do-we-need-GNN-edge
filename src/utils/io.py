"""Funciones sencillas de entrada/salida."""

import json
from pathlib import Path
from typing import Any


def ensure_directories(*paths: Path) -> None:
    """Crea directorios si no existen."""

    # Recorremos todas las rutas recibidas y las creamos si faltan.
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_json(data: dict[str, Any], path: Path) -> None:
    """Guarda un diccionario como JSON legible."""

    # Antes de guardar, nos aseguramos de que exista la carpeta destino.
    ensure_directories(path.parent)

    # Guardamos con indentacion para poder leer el fichero a mano.
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


def load_json(path: Path) -> dict[str, Any]:
    """Carga un JSON desde disco."""

    # Abrimos el fichero en modo lectura y lo convertimos a diccionario.
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
