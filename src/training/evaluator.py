import time
from io import BytesIO

import torch
from torch import nn

from config import ALPHA, DT, DX, GRID_SIZE
from training.trainer import diffusion_physics_target, model_forward


def evaluate_prediction_error(
    model:      nn.Module,
    x_input:    torch.Tensor,
    x_target:   torch.Tensor,
    edge_index: torch.Tensor,
) -> float:
    
    model.eval()
    total_mse = 0.0
    num_steps = x_input.shape[0]

    with torch.no_grad():
        for t in range(num_steps):
            x_pred     = model_forward(model, x_input[t], edge_index)
            total_mse += float(((x_pred - x_target[t]) ** 2).mean().item())

    return total_mse / num_steps


def evaluate_physics_violation(
    model:      nn.Module,
    x_input:    torch.Tensor,
    edge_index: torch.Tensor,
    alpha:      float = ALPHA,
    dt:         float = DT,
    dx:         float = DX,
    grid_size:  int   = GRID_SIZE,
) -> float:

    model.eval()
    total_phys = 0.0
    num_steps  = x_input.shape[0]

    with torch.no_grad():
        for t in range(num_steps):
            x_t   = x_input[t]                          # (N, 1)
            x_pred = model_forward(model, x_t, edge_index)
            x_physics = diffusion_physics_target(
                x_t,
                edge_index,
                grid_size=grid_size,
                alpha=alpha,
                dt=dt,
                dx=dx,
            )
            residual    = x_pred - x_physics
            total_phys += float((residual ** 2).mean().item())

    return total_phys / num_steps


def measure_inference_time(
    model:      nn.Module,
    x_input:    torch.Tensor,
    edge_index: torch.Tensor,
) -> float:

    model.eval()
    with torch.no_grad():
        start = time.perf_counter()
        _     = model_forward(model, x_input[0], edge_index)
        end   = time.perf_counter()
    return end - start


def measure_model_size_bytes(model: nn.Module) -> int:
    buffer = BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()
