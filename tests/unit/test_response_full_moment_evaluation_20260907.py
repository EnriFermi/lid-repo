import numpy as np
import pytest
import torch

from experiments.response_full_moment_evaluation_20260907 import (
    active_span_trace, check_oracle, load_model, noisy_metrics, physical_grid, readouts, select_scale,
)
from experiments.response_full_training_20260907 import posterior
from models.neural_fields import NeuralFieldConfig, ScaleConditionedNeuralField


def test_grid_contains_exact_oracle_anchors_and_selection_does_not_use_test():
    anchors = np.logspace(-2, 0, 9)
    grid = physical_grid(anchors)
    np.testing.assert_array_equal(grid[::3], anchors)
    values = np.full((25, 4), 4.)
    values[3] = values[4] = 2.
    chosen, risk = select_scale(values, 2., grid)
    assert chosen == 3 and risk[3] == 0.
    with pytest.raises(ValueError):
        physical_grid(anchors[::-1])


def test_noisy_moment_is_actual_full_minus_sampled_noise_energy():
    rng = np.random.default_rng(23)
    x, z, b = [rng.normal(size=(11, 3)) for _ in range(3)]
    r = rng.normal(size=11)
    scale = .17
    result = noisy_metrics(b, r, x, z, scale)
    full = r + ((b-x-scale*z)**2).sum(-1)/scale**2
    expected = ((full-(z*z).sum(-1))**2).mean()
    np.testing.assert_allclose(result["moment_mse_unscaled"], expected)
    np.testing.assert_allclose(result["moment_mse_div_D"], expected/3)
    np.testing.assert_allclose(result["original_denoising_loss"], ((b-x)**2).mean()/scale**2)


def test_oracle_requires_exact_queries_ids_and_normalization():
    q = np.arange(12, dtype=np.float32).reshape(4,3)
    fixture = {"query_ids": np.array([3,1]), "queries": torch.from_numpy(q[[3,1]]),
               "mean": torch.zeros(3), "rms": .2}
    oracle = {"query_ids": np.array([3,1]), "queries": q[[3,1]],
              "normalization_mean": np.zeros(3, dtype=np.float32), "normalization_scale": .2}
    check_oracle(fixture, oracle, q)
    wrong = dict(oracle, normalization_scale=.3)
    with pytest.raises(AssertionError):
        check_oracle(fixture, wrong, q)
    with pytest.raises(AssertionError):
        check_oracle(fixture, dict(oracle, query_ids=np.array([1,3])), q)


def test_load_preserves_sub_float32_weights_and_exact_trace():
    from dataclasses import asdict
    torch.manual_seed(6)
    cfg = NeuralFieldConfig(ambient_dim=3, hidden_dim=8, depth=2,
        condition_dim=8, fourier_features=4, zero_init_output=False)
    model = ScaleConditionedNeuralField(cfg).double()
    with torch.no_grad():
        model.output[-1].weight.add_(1e-10)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    payload = {"architecture": asdict(cfg), "models": {"control": state, "moment": state},
               "averages": {"control": state, "moment": state}}
    loaded = load_model(payload, "moment_average", "cpu")
    for key, value in loaded.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    q = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    actual = readouts(loaded, q, .2, 2)
    b = posterior(loaded, q, .2, "posterior")
    trace = sum(torch.autograd.grad(b[:,d].sum(), q, retain_graph=True)[0][:,d] for d in range(3))
    np.testing.assert_allclose(actual["response"], trace.detach().numpy(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual["posterior"], b.detach().numpy(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual["full"], actual["response"]+actual["correction"], rtol=0, atol=0)
    all_active = active_span_trace(loaded, q, .2, 3, 2)
    np.testing.assert_allclose(all_active, actual["response"], rtol=1e-10, atol=1e-10)
    active1 = active_span_trace(loaded, q, .2, 1, 3)
    first = torch.autograd.grad(b[:,0].sum(), q, retain_graph=True)[0][:,0]
    np.testing.assert_allclose(active1, first.detach().numpy(), rtol=1e-10, atol=1e-10)
