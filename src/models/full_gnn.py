import torch
from torch import nn
from torch_geometric.nn import GCNConv

from config import HIDDEN_DIM_FULL


class FullGNN(nn.Module):

    uses_graph = True

    def __init__(
        self,
        input_dim:   int   = 1,
        hidden_dim:  int   = HIDDEN_DIM_FULL,
        output_dim:  int   = 1,
        dropout:     float = 0.1,
    ) -> None:
        super().__init__()

        self.conv1   = GCNConv(input_dim,  hidden_dim)
        self.conv2   = GCNConv(hidden_dim, hidden_dim)
        self.conv3   = GCNConv(hidden_dim, output_dim)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:

        x = self.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        return self.conv3(x, edge_index)
