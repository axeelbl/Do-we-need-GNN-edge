from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from unittest import TestCase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.graph_builder import build_grid_graph  # noqa: E402
from data.heat_diffusion import (  # noqa: E402
    build_temporal_pairs,
    simulate_heat_diffusion_trajectories,
)
from run_sweep_experiments_v2 import (  # noqa: E402
    parse_float_list,
    parse_int_list,
    validate_experiment_config,
)


def valid_args(**overrides: object) -> Namespace:
    values = {
        "grid_size": 4,
        "trajectory_timesteps": 3,
        "train_trajectories": 2,
        "val_trajectories": 1,
        "test_trajectories": 1,
        "learning_rate": 1e-3,
        "dropout": 0.1,
        "data_alpha": 0.1,
        "pinn_alpha": 0.08,
        "dt": 1.0,
        "dx": 1.0,
        "min_epochs": 1,
        "patience": 1,
        "min_delta": 0.0,
        "inference_repeats": 1,
        "inference_warmup": 0,
        "limit_trainings": 0,
    }
    values.update(overrides)
    return Namespace(**values)


class ParsingTests(TestCase):
    def test_integer_range_is_inclusive(self) -> None:
        self.assertEqual(parse_int_list("40:42"), [40, 41, 42])

    def test_float_list_ignores_whitespace(self) -> None:
        self.assertEqual(parse_float_list("1, 0.5"), [1.0, 0.5])


class ValidationTests(TestCase):
    def validate(self, args: Namespace) -> None:
        validate_experiment_config(args, [5], [40], [16], [1.0], [0.0], [0.1])

    def test_valid_configuration_is_accepted(self) -> None:
        self.validate(valid_args())

    def test_invalid_grid_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "grid size"):
            self.validate(valid_args(grid_size=1))

    def test_invalid_data_fraction_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "data fractions"):
            validate_experiment_config(valid_args(), [5], [40], [16], [0.0], [0.0], [0.1])


class DataGenerationTests(TestCase):
    def test_independent_trajectory_shapes_and_pairs(self) -> None:
        data = simulate_heat_diffusion_trajectories(
            grid_size=4,
            num_timesteps=3,
            num_train_trajectories=2,
            num_val_trajectories=1,
            num_test_trajectories=1,
            seed=7,
        )

        self.assertEqual(tuple(data.train_states.shape), (2, 3, 16))
        inputs, targets = build_temporal_pairs(data.train_states)
        self.assertEqual(tuple(inputs.shape), (4, 16))
        self.assertEqual(tuple(targets.shape), (4, 16))

    def test_grid_graph_has_bidirectional_four_neighbour_edges(self) -> None:
        graph = build_grid_graph(grid_size=3)
        self.assertEqual(graph.num_nodes, 9)
        self.assertEqual(tuple(graph.edge_index.shape), (2, 24))
