import torch
from torch import nn
from torch.optim import Optimizer


def model_forward(
    model:      nn.Module,
    x:          torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    if getattr(model, "uses_graph", True):
        return model(x, edge_index)
    return model(x)


def train_one_epoch(
    model:      nn.Module,
    x_input:    torch.Tensor,
    x_target:   torch.Tensor,
    edge_index: torch.Tensor,
    optimizer:  Optimizer,
    loss_fn:    nn.Module,
    physics_lambda: float = 0.0,
    grid_size: int | None = None,
    alpha: float = 0.1,
    dt: float = 1.0,
    dx: float = 1.0,
) -> dict[str, float]:


    model.train()
    optimizer.zero_grad()

    num_steps       = x_input.shape[0]
    total_data_loss = torch.tensor(0.0, device=x_input.device)
    total_phys_loss = torch.tensor(0.0, device=x_input.device)

    for t in range(num_steps):
        x_pred      = model_forward(model, x_input[t], edge_index)
        total_data_loss = total_data_loss + loss_fn(x_pred, x_target[t])
        if physics_lambda > 0.0:
            x_physics = diffusion_physics_target(
                x_input[t],
                edge_index,
                grid_size=grid_size,
                alpha=alpha,
                dt=dt,
                dx=dx,
            )
            total_phys_loss = total_phys_loss + loss_fn(x_pred, x_physics)

    mean_data_loss = total_data_loss / num_steps
    mean_phys_loss = total_phys_loss / num_steps
    total_loss = mean_data_loss + physics_lambda * mean_phys_loss
    total_loss.backward()
    optimizer.step()

    return {
        "data_loss":  float(mean_data_loss.item()),
        "physics_loss": float(mean_phys_loss.item()),
        "total_loss": float(total_loss.item()),
    }


def diffusion_physics_target(
    x_t: torch.Tensor,
    edge_index: torch.Tensor,
    grid_size: int | None = None,
    alpha: float = 0.1,
    dt: float = 1.0,
    dx: float = 1.0,
) -> torch.Tensor:
    """One explicit heat-diffusion step on the graph.

    The grid graph is 4-connected and directed both ways. For each target node,
    the scatter accumulates sum(neighbour - center), i.e. the discrete Laplacian.
    Dirichlet boundary conditions are enforced by clamping border nodes to zero.
    """
    r = alpha * dt / (dx ** 2)
    src, tgt = edge_index[0], edge_index[1]
    neighbour_minus_center = x_t[src] - x_t[tgt]
    laplacian = torch.zeros_like(x_t)
    laplacian.scatter_add_(0, tgt.unsqueeze(1).expand_as(neighbour_minus_center), neighbour_minus_center)
    x_next = x_t + r * laplacian

    if grid_size is not None:
        mask = _boundary_mask(grid_size, device=x_t.device)
        x_next = x_next.clone()
        x_next[mask] = 0.0

    return x_next


def _boundary_mask(grid_size: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(grid_size * grid_size, device=device)
    rows = idx // grid_size
    cols = idx % grid_size
    return (rows == 0) | (rows == grid_size - 1) | (cols == 0) | (cols == grid_size - 1)
