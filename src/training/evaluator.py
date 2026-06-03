import time

import torch
from torch import nn

from config import ALPHA, DT, DX
from training.trainer import model_forward


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
) -> float:

    model.eval()
    r          = alpha * dt / (dx ** 2)
    total_phys = 0.0
    num_steps  = x_input.shape[0]

    with torch.no_grad():
        for t in range(num_steps):
            x_t   = x_input[t]                          # (N, 1)
            x_pred = model_forward(model, x_t, edge_index)

            src, tgt = edge_index[0], edge_index[1]
            diff = x_t[src] - x_t[tgt]
            laplacian = torch.zeros_like(x_t)
            laplacian.scatter_add_(0, tgt.unsqueeze(1).expand_as(diff), diff)

            x_physics   = x_t - r * laplacian
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
