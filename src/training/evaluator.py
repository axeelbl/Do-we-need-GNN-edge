from __future__ import annotations

import time
from io import BytesIO

import torch
from torch import nn

from config import ALPHA, DT, DX, GRID_SIZE
from training.trainer import diffusion_physics_target, model_forward


def evaluate_prediction_error(
    model: nn.Module,
    x_input: torch.Tensor,
    x_target: torch.Tensor,
    edge_index: torch.Tensor,
) -> float:
    model.eval()
    total_mse = 0.0
    num_steps = x_input.shape[0]

    with torch.no_grad():
        for t in range(num_steps):
            x_pred = model_forward(model, x_input[t], edge_index)
            total_mse += float(((x_pred - x_target[t]) ** 2).mean().item())

    return total_mse / max(num_steps, 1)


def evaluate_physics_violation(
    model: nn.Module,
    x_input: torch.Tensor,
    edge_index: torch.Tensor,
    alpha: float = ALPHA,
    dt: float = DT,
    dx: float = DX,
    grid_size: int = GRID_SIZE,
) -> float:
    """Mean residual MSE against the physics assumed by the PINN.

    In the corrected experiments this alpha is normally pinn_alpha, while the
    data may have been generated with a different data_alpha and/or a reaction
    term. Therefore this metric is no longer automatically equal to test MSE.
    """

    model.eval()
    total_phys = 0.0
    num_steps = x_input.shape[0]

    with torch.no_grad():
        for t in range(num_steps):
            x_t = x_input[t]
            x_pred = model_forward(model, x_t, edge_index)
            x_physics = diffusion_physics_target(
                x_t,
                edge_index,
                grid_size=grid_size,
                alpha=alpha,
                dt=dt,
                dx=dx,
            )
            residual = x_pred - x_physics
            total_phys += float((residual ** 2).mean().item())

    return total_phys / max(num_steps, 1)


def _cuda_sync_if_needed(x: torch.Tensor | None = None) -> None:
    if torch.cuda.is_available():
        if x is None or x.device.type == "cuda":
            torch.cuda.synchronize()


def measure_inference_time_stats_ms(
    model: nn.Module,
    x_input: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    repeats: int = 1000,
    warmup: int = 50,
) -> dict[str, float]:
    """Robust inference timing over many single-step forward passes."""

    model.eval()
    repeats = max(1, int(repeats))
    warmup = max(0, int(warmup))
    x0 = x_input[0]

    with torch.no_grad():
        for _ in range(warmup):
            _ = model_forward(model, x0, edge_index)
        _cuda_sync_if_needed(x0)

        times_ms: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            _ = model_forward(model, x0, edge_index)
            _cuda_sync_if_needed(x0)
            end = time.perf_counter()
            times_ms.append((end - start) * 1000.0)

    t = torch.tensor(times_ms, dtype=torch.float64)
    return {
        "mean_ms": float(t.mean().item()),
        "std_ms": float(t.std(unbiased=True).item()) if repeats > 1 else 0.0,
        "min_ms": float(t.min().item()),
        "max_ms": float(t.max().item()),
        "repeats": float(repeats),
    }


def measure_inference_time(
    model: nn.Module,
    x_input: torch.Tensor,
    edge_index: torch.Tensor,
) -> float:
    """Backwards-compatible helper returning mean seconds."""
    stats = measure_inference_time_stats_ms(model, x_input, edge_index, repeats=100, warmup=10)
    return stats["mean_ms"] / 1000.0


def measure_model_size_bytes(model: nn.Module) -> int:
    buffer = BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()
