import torch
from torch import nn

from config import HIDDEN_DIM_FULL


class MLPBaseline(nn.Module):

    uses_graph = False

    def __init__(
        self,
        input_dim:  int = 1,
        hidden_dim: int = HIDDEN_DIM_FULL,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:

        return self.net(x)
