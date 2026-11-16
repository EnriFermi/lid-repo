"""Small validity checks for the isolated rendered convergence canary."""

import torch

from experiments.spiral_spaghetti_convergence import check_trace, exact_trace, posterior
from models.neural_fields import NeuralFieldConfig, ScaleConditionedNeuralField


def test_linear_trace() -> None:
    assert check_trace()["linear_trace_max_error"] < 1e-12


def test_batched_trace_matches_coordinate_loop() -> None:
    torch.manual_seed(12)
    model = ScaleConditionedNeuralField(NeuralFieldConfig(
        ambient_dim=5, hidden_dim=8, depth=1, condition_dim=4,
        fourier_features=2, zero_init_output=False,
    )).double()
    points = torch.randn(3, 5, dtype=torch.float64)
    value, trace = exact_trace(model, points, .3)
    x = points.clone().requires_grad_(True)
    y = posterior(model, x, .3)
    reference = torch.stack([
        torch.autograd.grad(y[:, i].sum(), x, retain_graph=True)[0][:, i]
        for i in range(5)
    ]).sum(0)
    torch.testing.assert_close(value, y)
    torch.testing.assert_close(trace, reference)


def test_trace_across_query_and_coordinate_chunks() -> None:
    class Linear(torch.nn.Module):
        def forward(self, inputs, condition):
            return inputs * torch.arange(1, 38, dtype=inputs.dtype)

    points = torch.randn(11, 37, dtype=torch.float64)
    _, trace = exact_trace(Linear(), points, .25)
    expected = torch.full((11,), sum(range(1, 38)) / 1.25, dtype=torch.float64)
    torch.testing.assert_close(trace, expected)
