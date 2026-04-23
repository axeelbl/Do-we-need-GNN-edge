"""Control de semillas para experimentos reproducibles."""

import os
import random


def set_seed(seed: int) -> None:
    """Fija semillas en librerias disponibles."""

    # Controla operaciones internas que dependen del hash de Python.
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Semilla para la libreria estandar random.
    random.seed(seed)

    try:
        # Semilla para numpy, si esta instalado.
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        # El proyecto puede seguir si numpy no esta disponible en este entorno.
        pass

    try:
        # Semilla para PyTorch, si esta instalado.
        import torch

        torch.manual_seed(seed)

        # Si hay GPU, tambien fijamos la semilla de CUDA.
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # Permite importar esta utilidad incluso antes de instalar torch.
        pass
