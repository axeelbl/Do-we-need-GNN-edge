"""Configuracion centralizada del pipeline de datos."""

from dataclasses import dataclass, field
from pathlib import Path


# Ruta raiz del proyecto. Se calcula a partir de este archivo.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Carpeta para los datos descargados
DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "src" / "results"

# Configuracion del dataset CIFAR-10
TRAIN_SUBSET_SIZE = 1000
TEST_SUBSET_SIZE = 300
NUM_CLASSES = 10

# Cada imagen CIFAR-10 tiene forma 3x32x32. Al hacer flatten queda en 3072.
IMAGE_FEATURE_DIM = 3 * 32 * 32

# Dimensiones que usaran los modelos GNN
INPUT_DIM = IMAGE_FEATURE_DIM
FULL_GNN_HIDDEN_DIM = 128
TINY_GNN_HIDDEN_DIM = 32
DROPOUT = 0.3

# Parametros para construir el grafo de similitud
K_NEIGHBORS = 8
NORMALIZE_FEATURES = True

# Semilla fija para que los subconjuntos y resultados sean reproducibles
RANDOM_SEED = 42

# Parametros basicos de entrenamiento
EPOCHS = 1
LEARNING_RATE = 1e-3
DEVICE = "cpu"


@dataclass(frozen=True)
class DataConfig:
    """Configuracion del dataset para mantener imports existentes."""

    # Nombre del dataset usado en el proyecto
    dataset_name: str = "CIFAR-10"

    # Carpeta donde torchvision guarda/lee CIFAR-10
    raw_dir: Path = DATA_DIR

    # Tamanos de los subconjuntos usados para entrenar y evaluar
    train_subset: int = TRAIN_SUBSET_SIZE
    test_subset: int = TEST_SUBSET_SIZE


@dataclass(frozen=True)
class GraphConfig:
    """Configuracion basica del grafo."""

    # Numero de vecinos que se conectan por cada nodo
    k_neighbors: int = K_NEIGHBORS

    # Dimension del vector de features de cada imagen
    feature_dim: int = IMAGE_FEATURE_DIM


@dataclass(frozen=True)
class ModelConfig:
    """Configuracion comun de los modelos GNN."""

    # Dimension de entrada de cada nodo
    input_dim: int = INPUT_DIM

    # Numero de clases de CIFAR-10
    num_classes: int = NUM_CLASSES

    # Capacidad del modelo grande
    full_hidden_dim: int = FULL_GNN_HIDDEN_DIM

    # Capacidad del modelo pequeno
    tiny_hidden_dim: int = TINY_GNN_HIDDEN_DIM

    # Probabilidad de apagar neuronas durante entrenamiento
    dropout: float = DROPOUT


@dataclass(frozen=True)
class TrainingConfig:
    """Configuracion minima para la fase de entrenamiento posterior."""

    # En esta primera comparacion solo se entrena una epoca
    epochs: int = EPOCHS

    # Tasa de aprendizaje del optimizador Adam
    learning_rate: float = LEARNING_RATE

    # Semilla usada por el pipeline
    seed: int = RANDOM_SEED

    # Dispositivo por defecto: CPU para maxima compatibilidad
    device: str = DEVICE


@dataclass(frozen=True)
class AppConfig:
    """Configuracion agrupada para utilidades existentes."""

    # Agrupa la configuracion por areas para acceder de forma ordenada
    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Fichero donde se guardan las metricas finales
    metrics_path: Path = RESULTS_DIR / "metrics.json"


# Objeto global de configuracion para modulos que prefieran usar dataclasses
CONFIG = AppConfig()
