"""Configuracion centralizada del pipeline de datos."""

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "src" / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

TRAIN_SUBSET_SIZE = 1000
TEST_SUBSET_SIZE = 300
NUM_CLASSES = 10
IMAGES_PER_NODE = 5

RUN_MODE = "resource_efficiency"

FULL_GNN_TRAIN_SUBSET_SIZE: int | None = None
FULL_GNN_TEST_SUBSET_SIZE: int | None = None

TINY_GNN_TRAIN_SUBSET_SIZE = TRAIN_SUBSET_SIZE
TINY_GNN_TEST_SUBSET_SIZE = TEST_SUBSET_SIZE

CONTROLLED_TRAIN_SUBSET_SIZE = TRAIN_SUBSET_SIZE
CONTROLLED_TEST_SUBSET_SIZE = TEST_SUBSET_SIZE

IMAGE_FEATURE_DIM = 3 * 32 * 32

INPUT_DIM = IMAGE_FEATURE_DIM
FULL_GNN_HIDDEN_DIM = 128
TINY_GNN_HIDDEN_DIM = 32
DROPOUT = 0.3

K_NEIGHBORS = 8
NORMALIZE_FEATURES = True

RANDOM_SEED = 42

EPOCHS = 3
LEARNING_RATE = 1e-3
DEVICE = "cpu"

SWEEP_RUN_MODE = "resource_efficiency"
SWEEP_K_NEIGHBORS = [5, 8]
SWEEP_EPOCHS = [1, 3]
SWEEP_IMAGES_PER_NODE = [1, 5, 10]
SWEEP_RESULTS_DIR = RESULTS_DIR / "config_sweep"


@dataclass(frozen=True)
class DataConfig:
    """Configuracion del dataset para mantener imports existentes."""

    dataset_name: str = "CIFAR-10"
    raw_dir: Path = DATA_DIR
    train_subset: int = TRAIN_SUBSET_SIZE
    test_subset: int = TEST_SUBSET_SIZE
    images_per_node: int = IMAGES_PER_NODE
    run_mode: str = RUN_MODE

    full_gnn_train_subset: int | None = FULL_GNN_TRAIN_SUBSET_SIZE
    full_gnn_test_subset: int | None = FULL_GNN_TEST_SUBSET_SIZE
    tiny_gnn_train_subset: int = TINY_GNN_TRAIN_SUBSET_SIZE
    tiny_gnn_test_subset: int = TINY_GNN_TEST_SUBSET_SIZE

    controlled_train_subset: int = CONTROLLED_TRAIN_SUBSET_SIZE
    controlled_test_subset: int = CONTROLLED_TEST_SUBSET_SIZE


@dataclass(frozen=True)
class GraphConfig:
    """Configuracion basica del grafo."""

    k_neighbors: int = K_NEIGHBORS
    feature_dim: int = IMAGE_FEATURE_DIM


@dataclass(frozen=True)
class ModelConfig:
    """Configuracion comun de los modelos GNN."""

    input_dim: int = INPUT_DIM
    num_classes: int = NUM_CLASSES
    full_hidden_dim: int = FULL_GNN_HIDDEN_DIM
    tiny_hidden_dim: int = TINY_GNN_HIDDEN_DIM
    dropout: float = DROPOUT


@dataclass(frozen=True)
class TrainingConfig:
    """Configuracion minima del entrenamiento."""

    epochs: int = EPOCHS
    learning_rate: float = LEARNING_RATE
    seed: int = RANDOM_SEED
    device: str = DEVICE
    sweep_run_mode: str = SWEEP_RUN_MODE
    sweep_k_neighbors: list[int] = field(default_factory=lambda: SWEEP_K_NEIGHBORS)
    sweep_epochs: list[int] = field(default_factory=lambda: SWEEP_EPOCHS)
    sweep_images_per_node: list[int] = field(default_factory=lambda: SWEEP_IMAGES_PER_NODE)


@dataclass(frozen=True)
class AppConfig:
    """Configuracion agrupada para utilidades existentes."""

    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    metrics_path: Path = RESULTS_DIR / "metrics.json"
    plots_dir: Path = PLOTS_DIR
    sweep_results_dir: Path = SWEEP_RESULTS_DIR


CONFIG = AppConfig()
