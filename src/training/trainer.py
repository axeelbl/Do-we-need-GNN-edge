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
) -> dict[str, float]:


    model.train()
    optimizer.zero_grad()

    num_steps       = x_input.shape[0]
    total_data_loss = torch.tensor(0.0, device=x_input.device)

    for t in range(num_steps):
        x_pred      = model_forward(model, x_input[t], edge_index)
        total_data_loss = total_data_loss + loss_fn(x_pred, x_target[t])

    mean_data_loss = total_data_loss / num_steps
    mean_data_loss.backward()
    optimizer.step()

    return {
        "data_loss":  float(mean_data_loss.item()),
        "total_loss": float(mean_data_loss.item()),
    }
