import torch
from torch import nn
from torch_geometric.nn import GCNConv

from config import HIDDEN_DIM_TINY


class TinyGNN(nn.Module):
    uses_graph = True

    def __init__(
        self,
        input_dim:  int = 1,
        hidden_dim: int = HIDDEN_DIM_TINY,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        self.conv1 = GCNConv(input_dim,  hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.relu  = nn.ReLU()

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:          (N, 1) temperatura actual de cada node.
            edge_index: (2, E) arestes del graf de veïnatge.
        Returns:
            (N, 1) temperatura predicha.
        """
        x = self.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)
