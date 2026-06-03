from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "src" / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"

GRID_SIZE      = 32    
NUM_TIMESTEPS  = 200   
TRAIN_STEPS    = 120
VAL_STEPS      = 30
TEST_STEPS     = 50
ALPHA          = 0.1   
DT             = 1.0   
DX             = 1.0   

K_NEIGHBORS    = 4   

HIDDEN_DIM_FULL = 64   
HIDDEN_DIM_TINY = 16 

EPOCHS         = 300
LEARNING_RATE  = 1e-3
PHYSICS_LAMBDA = 0.1
RANDOM_SEED    = 42
DEVICE         = "cpu"

SWEEP_HIDDEN_DIMS = [32, 16, 8, 4]
SWEEP_DATA_FRACTIONS = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
SWEEP_NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1, 0.2]
SWEEP_RESULTS_DIR = RESULTS_DIR / "config_sweep"


@dataclass(frozen=True)
class GridConfig:
    grid_size:     int   = GRID_SIZE
    num_timesteps: int   = NUM_TIMESTEPS
    train_steps:   int   = TRAIN_STEPS
    val_steps:     int   = VAL_STEPS
    test_steps:    int   = TEST_STEPS
    alpha:         float = ALPHA
    dt:            float = DT
    dx:            float = DX


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim_full: int = HIDDEN_DIM_FULL
    hidden_dim_tiny: int = HIDDEN_DIM_TINY
    input_dim:       int = 1   # temperatura a cada node
    output_dim:      int = 1   # temperatura predicha


@dataclass(frozen=True)
class TrainingConfig:
    epochs:        int   = EPOCHS
    learning_rate: float = LEARNING_RATE
    physics_lambda: float = PHYSICS_LAMBDA
    seed:          int   = RANDOM_SEED
    device:        str   = DEVICE


@dataclass(frozen=True)
class AppConfig:
    grid:             GridConfig     = field(default_factory=GridConfig)
    model:            ModelConfig    = field(default_factory=ModelConfig)
    training:         TrainingConfig = field(default_factory=TrainingConfig)
    results_dir:      Path           = RESULTS_DIR
    plots_dir:        Path           = PLOTS_DIR
    sweep_results_dir: Path          = SWEEP_RESULTS_DIR


CONFIG = AppConfig()
