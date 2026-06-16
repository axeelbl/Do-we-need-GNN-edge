
from dataclasses import dataclass

import torch

from config import ALPHA, DT, DX, GRID_SIZE, NUM_TIMESTEPS, RANDOM_SEED, TEST_STEPS, TRAIN_STEPS, VAL_STEPS


@dataclass(frozen=True)
class DiffusionData:
    states:       torch.Tensor 
    train_states: torch.Tensor   
    val_states:   torch.Tensor
    test_states:  torch.Tensor   
    grid_size:    int
    alpha:        float
    dt:           float
    dx:           float


def simulate_heat_diffusion(
    grid_size:     int   = GRID_SIZE,
    num_timesteps: int   = NUM_TIMESTEPS,
    train_steps:   int   = TRAIN_STEPS,
    val_steps:     int   = VAL_STEPS,
    test_steps:    int   = TEST_STEPS,
    alpha:         float = ALPHA,
    dt:            float = DT,
    dx:            float = DX,
    seed:          int   = RANDOM_SEED,
) -> DiffusionData:
    
    generator = torch.Generator().manual_seed(seed)

    grid = _gaussian_initial_condition(grid_size, generator)

    states: list[torch.Tensor] = [grid.reshape(-1).clone()]
    for _ in range(num_timesteps - 1):
        grid = _diffusion_step(grid, alpha, dt, dx)
        states.append(grid.reshape(-1).clone())

    all_states  = torch.stack(states)
    min_required = train_steps + val_steps + test_steps
    if min_required > num_timesteps:
        raise ValueError(
            "train_steps + val_steps + test_steps must be <= num_timesteps "
            f"({min_required} > {num_timesteps})"
        )

    train_end = train_steps
    val_end = train_end + val_steps
    test_end = val_end + test_steps

    return DiffusionData(
        states       = all_states,
        train_states = all_states[:train_steps],
        val_states   = all_states[train_end:val_end],
        test_states  = all_states[val_end:test_end],
        grid_size    = grid_size,
        alpha        = alpha,
        dt           = dt,
        dx           = dx,
    )


def build_temporal_pairs(
    states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return states[:-1], states[1:]

def add_gaussian_noise(
    states: torch.Tensor,
    sigma:  float,
    seed:   int = RANDOM_SEED,
) -> torch.Tensor:
    
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(states.shape, generator=generator) * sigma
    return (states + noise).clamp(min=0.0)


def select_data_fraction(
    x_input: torch.Tensor,
    x_target: torch.Tensor,
    fraction: float,
    seed: int = RANDOM_SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Selects a deterministic subset of temporal training pairs."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"data_fraction must be in (0, 1], got {fraction}")

    num_pairs = x_input.shape[0]
    keep = max(1, int(round(num_pairs * fraction)))
    if keep >= num_pairs:
        return x_input, x_target

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_pairs, generator=generator)[:keep].sort().values
    return x_input[indices], x_target[indices]

def _gaussian_initial_condition(
    grid_size: int,
    generator: torch.Generator,
) -> torch.Tensor:

    coords = torch.arange(grid_size, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")

    cx, cy = grid_size / 2.0, grid_size / 2.0
    sigma  = grid_size / 6.0

    gaussian = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    noise    = torch.rand(grid_size, grid_size, generator=generator) * 0.1
    grid     = gaussian + noise

    # Condicions de contorn Dirichlet
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 0.0
    return grid


def _diffusion_step(
    grid:  torch.Tensor,
    alpha: float,
    dt:    float,
    dx:    float,
) -> torch.Tensor:

    r = alpha * dt / (dx ** 2)

    laplacian = (
        torch.roll(grid, -1, dims=0)
        + torch.roll(grid,  1, dims=0)
        + torch.roll(grid, -1, dims=1)
        + torch.roll(grid,  1, dims=1)
        - 4.0 * grid
    )

    new_grid = grid + r * laplacian
    new_grid[0, :] = new_grid[-1, :] = new_grid[:, 0] = new_grid[:, -1] = 0.0
    return new_grid
