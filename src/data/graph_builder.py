import torch
from torch_geometric.data import Data

from config import GRID_SIZE


def build_grid_graph(grid_size: int = GRID_SIZE) -> Data:
    edge_index = _build_4connected_edges(grid_size)

    return Data(
        edge_index=edge_index,
        num_nodes=grid_size * grid_size,
    )


def _build_4connected_edges(grid_size: int) -> torch.Tensor:

    sources: list[int] = []
    targets: list[int] = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for row in range(grid_size):
        for col in range(grid_size):
            node = row * grid_size + col
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    sources.append(node)
                    targets.append(nr * grid_size + nc)

    return torch.tensor([sources, targets], dtype=torch.long)
