from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from config import ALPHA, DT, DX, GRID_SIZE, NUM_TIMESTEPS, RANDOM_SEED, TEST_STEPS, TRAIN_STEPS, VAL_STEPS


@dataclass(frozen=True)
class DiffusionData:
    states: torch.Tensor
    train_states: torch.Tensor
    val_states: torch.Tensor
    test_states: torch.Tensor
    grid_size: int
    alpha: float
    dt: float
    dx: float


@dataclass(frozen=True)
class TrajectoryDiffusionData:
    """Independent heat-diffusion trajectories.

    Shapes:
        train_states: (num_train_trajectories_kept, T, N)
        val_states:   (num_val_trajectories, T, N)
        test_states:  (num_test_trajectories, T, N)
    """

    train_states: torch.Tensor
    val_states: torch.Tensor
    test_states: torch.Tensor
    grid_size: int
    data_alpha: float
    pinn_alpha: float
    reaction_mu: float
    dt: float
    dx: float
    num_train_trajectories_total: int
    num_train_trajectories_kept: int
    num_val_trajectories: int
    num_test_trajectories: int
    data_fraction: float
    initial_conditions: list[dict[str, Any]]


# -----------------------------------------------------------------------------
# Backwards-compatible single-trajectory simulator.
# Kept because main.py / old experiments may still import this function.
# -----------------------------------------------------------------------------
def simulate_heat_diffusion(
    grid_size: int = GRID_SIZE,
    num_timesteps: int = NUM_TIMESTEPS,
    train_steps: int = TRAIN_STEPS,
    val_steps: int = VAL_STEPS,
    test_steps: int = TEST_STEPS,
    alpha: float = ALPHA,
    dt: float = DT,
    dx: float = DX,
    seed: int = RANDOM_SEED,
    random_initial_condition: bool = True,
    reaction_mu: float = 0.0,
) -> DiffusionData:
    """Simulate one trajectory.

    NOTE: for real train/val/test generalisation, prefer
    simulate_heat_diffusion_trajectories(), which splits by full independent
    trajectories instead of consecutive time windows from the same trajectory.
    """

    generator = torch.Generator().manual_seed(seed)
    if random_initial_condition:
        grid, _ = _random_gaussian_initial_condition(grid_size, generator)
    else:
        grid = _centered_gaussian_initial_condition(grid_size, generator)

    states: list[torch.Tensor] = [grid.reshape(-1).clone()]
    for _ in range(num_timesteps - 1):
        grid = _diffusion_reaction_step(grid, alpha=alpha, dt=dt, dx=dx, reaction_mu=reaction_mu)
        states.append(grid.reshape(-1).clone())

    all_states = torch.stack(states)
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
        states=all_states,
        train_states=all_states[:train_steps],
        val_states=all_states[train_end:val_end],
        test_states=all_states[val_end:test_end],
        grid_size=grid_size,
        alpha=alpha,
        dt=dt,
        dx=dx,
    )


# -----------------------------------------------------------------------------
# New recommended data generator for the corrected TFG experiments.
# -----------------------------------------------------------------------------
def simulate_heat_diffusion_trajectories(
    *,
    grid_size: int = GRID_SIZE,
    num_timesteps: int = 60,
    num_train_trajectories: int = 12,
    num_val_trajectories: int = 4,
    num_test_trajectories: int = 4,
    data_fraction: float = 1.0,
    data_alpha: float = ALPHA,
    pinn_alpha: float | None = None,
    reaction_mu: float = 0.02,
    dt: float = DT,
    dx: float = DX,
    seed: int = RANDOM_SEED,
) -> TrajectoryDiffusionData:
    """Generate independent trajectories and split by trajectory.

    Fixes the main methodological issues:
      1. The data dynamics can include controlled physics mismatch:
         data_alpha != pinn_alpha and/or reaction_mu > 0 in the data while the
         PINN still enforces pure diffusion.
      2. Train/validation/test are independent trajectories, not consecutive
         windows from the same temporal sequence.
      3. Every trajectory has a random Gaussian source: position, amplitude and
         width are sampled independently.
      4. data_fraction selects whole training trajectories, not individual
         timesteps.
    """

    if not 0.0 < data_fraction <= 1.0:
        raise ValueError(f"data_fraction must be in (0, 1], got {data_fraction}")
    if num_timesteps < 2:
        raise ValueError("num_timesteps must be >= 2")
    if min(num_train_trajectories, num_val_trajectories, num_test_trajectories) < 1:
        raise ValueError("All trajectory counts must be >= 1")

    pinn_alpha = data_alpha if pinn_alpha is None else pinn_alpha
    total_trajectories = num_train_trajectories + num_val_trajectories + num_test_trajectories
    generator = torch.Generator().manual_seed(seed)

    trajectories: list[torch.Tensor] = []
    initial_conditions: list[dict[str, Any]] = []
    for trajectory_id in range(total_trajectories):
        states, params = _simulate_one_random_trajectory(
            grid_size=grid_size,
            num_timesteps=num_timesteps,
            data_alpha=data_alpha,
            reaction_mu=reaction_mu,
            dt=dt,
            dx=dx,
            generator=generator,
        )
        params["trajectory_id"] = trajectory_id
        trajectories.append(states)
        initial_conditions.append(params)

    all_states = torch.stack(trajectories)  # (B, T, N)

    train_all = all_states[:num_train_trajectories]
    val_states = all_states[num_train_trajectories:num_train_trajectories + num_val_trajectories]
    test_states = all_states[num_train_trajectories + num_val_trajectories:]

    # Subsample full trajectories for data scarcity experiments.
    keep = max(1, int(round(num_train_trajectories * data_fraction)))
    if keep < num_train_trajectories:
        perm_generator = torch.Generator().manual_seed(seed + 10_007)
        kept_idx = torch.randperm(num_train_trajectories, generator=perm_generator)[:keep].sort().values
        train_states = train_all[kept_idx]
    else:
        train_states = train_all

    return TrajectoryDiffusionData(
        train_states=train_states,
        val_states=val_states,
        test_states=test_states,
        grid_size=grid_size,
        data_alpha=float(data_alpha),
        pinn_alpha=float(pinn_alpha),
        reaction_mu=float(reaction_mu),
        dt=float(dt),
        dx=float(dx),
        num_train_trajectories_total=int(num_train_trajectories),
        num_train_trajectories_kept=int(train_states.shape[0]),
        num_val_trajectories=int(num_val_trajectories),
        num_test_trajectories=int(num_test_trajectories),
        data_fraction=float(data_fraction),
        initial_conditions=initial_conditions,
    )


def build_temporal_pairs(states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one-step pairs from either one trajectory or multiple trajectories.

    Accepted shapes:
        (T, N)    -> returns (T-1, N)
        (B, T, N) -> returns (B*(T-1), N)
    """

    if states.dim() == 2:
        return states[:-1], states[1:]
    if states.dim() == 3:
        b, t, n = states.shape
        return states[:, :-1, :].reshape(b * (t - 1), n), states[:, 1:, :].reshape(b * (t - 1), n)
    raise ValueError(f"states must have shape (T, N) or (B, T, N), got {tuple(states.shape)}")


def add_gaussian_noise(
    states: torch.Tensor,
    sigma: float,
    seed: int = RANDOM_SEED,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(states.shape, generator=generator, device=states.device) * sigma
    return (states + noise).clamp(min=0.0)


def select_data_fraction(
    x_input: torch.Tensor,
    x_target: torch.Tensor,
    fraction: float,
    seed: int = RANDOM_SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Legacy helper: selects temporal pairs.

    Kept for compatibility with old scripts. The corrected data-scarcity
    experiment should use simulate_heat_diffusion_trajectories(data_fraction=...),
    because that subsamples full trajectories instead of timesteps.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"data_fraction must be in (0, 1], got {fraction}")

    num_pairs = x_input.shape[0]
    keep = max(1, int(round(num_pairs * fraction)))
    if keep >= num_pairs:
        return x_input, x_target

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_pairs, generator=generator)[:keep].sort().values
    return x_input[indices], x_target[indices]


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------
def _simulate_one_random_trajectory(
    *,
    grid_size: int,
    num_timesteps: int,
    data_alpha: float,
    reaction_mu: float,
    dt: float,
    dx: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, Any]]:
    grid, params = _random_gaussian_initial_condition(grid_size, generator)
    states: list[torch.Tensor] = [grid.reshape(-1).clone()]
    for _ in range(num_timesteps - 1):
        grid = _diffusion_reaction_step(grid, alpha=data_alpha, dt=dt, dx=dx, reaction_mu=reaction_mu)
        states.append(grid.reshape(-1).clone())
    return torch.stack(states), params


def _random_gaussian_initial_condition(
    grid_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, Any]]:
    coords = torch.arange(grid_size, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")

    # Keep the heat source away from the boundary so boundary conditions do not
    # dominate all trajectories.
    margin = max(2.0, grid_size * 0.15)
    low = margin
    high = grid_size - 1 - margin

    cx = _uniform(generator, low, high)
    cy = _uniform(generator, low, high)
    amplitude = _uniform(generator, 0.6, 1.4)
    sigma = _uniform(generator, grid_size / 12.0, grid_size / 4.5)
    noise_amplitude = _uniform(generator, 0.0, 0.05)

    gaussian = amplitude * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    noise = torch.rand(grid_size, grid_size, generator=generator) * noise_amplitude
    grid = gaussian + noise
    _apply_dirichlet_boundary_(grid)

    params = {
        "cx": float(cx),
        "cy": float(cy),
        "amplitude": float(amplitude),
        "sigma": float(sigma),
        "noise_amplitude": float(noise_amplitude),
    }
    return grid, params


def _centered_gaussian_initial_condition(
    grid_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    coords = torch.arange(grid_size, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")

    cx, cy = grid_size / 2.0, grid_size / 2.0
    sigma = grid_size / 6.0

    gaussian = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    noise = torch.rand(grid_size, grid_size, generator=generator) * 0.1
    grid = gaussian + noise
    _apply_dirichlet_boundary_(grid)
    return grid


def _diffusion_reaction_step(
    grid: torch.Tensor,
    alpha: float,
    dt: float,
    dx: float,
    reaction_mu: float = 0.0,
) -> torch.Tensor:
    """Explicit diffusion step plus optional small reaction term in the data.

    The PINN can still enforce pure diffusion, so reaction_mu > 0 creates a
    controlled physics mismatch: the labels come from diffusion+reaction, while
    L_physics can be computed from diffusion only.
    """

    r = alpha * dt / (dx ** 2)
    if r > 0.25:
        raise ValueError(
            f"Unstable explicit diffusion setting: alpha*dt/dx^2 = {r:.3f}; use <= 0.25"
        )

    laplacian = (
        torch.roll(grid, -1, dims=0)
        + torch.roll(grid, 1, dims=0)
        + torch.roll(grid, -1, dims=1)
        + torch.roll(grid, 1, dims=1)
        - 4.0 * grid
    )

    # Small nonlinear source/decay term. Positive mu increases heat in the
    # interior for values below 1. This term is intentionally absent from the
    # PINN physics residual.
    reaction = reaction_mu * grid * (1.0 - grid)
    new_grid = grid + r * laplacian + dt * reaction
    new_grid = new_grid.clamp(min=0.0)
    _apply_dirichlet_boundary_(new_grid)
    return new_grid


def _apply_dirichlet_boundary_(grid: torch.Tensor) -> None:
    grid[0, :] = 0.0
    grid[-1, :] = 0.0
    grid[:, 0] = 0.0
    grid[:, -1] = 0.0


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    return float(low + (high - low) * torch.rand((), generator=generator).item())
