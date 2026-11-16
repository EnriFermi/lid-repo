import copy
from dataclasses import replace

import pytest
import torch

from experiments.fm_mechanism_fast_trace import fast_value_and_trace
from experiments.fm_mechanism_pilot import PilotConfig, canonical_posterior, train_pilot
from experiments.fm_scheduler_cause_gauge import (
    apply_condition_gauge, independent_adam, optimizer_step, original_loss,
    parameter_factor, run,
)
from models.neural_fields import NeuralFieldConfig, ScaleConditionedNeuralField


def fixture():
    torch.manual_seed(504)
    field = ScaleConditionedNeuralField(NeuralFieldConfig(
        ambient_dim=4, hidden_dim=12, depth=2, condition_dim=8,
        fourier_features=4, condition_transform="linear", zero_init_output=False)).double()
    optimizer = torch.optim.AdamW(field.parameters(), lr=.001, eps=1e-6, weight_decay=.0001)
    x = torch.randn(11, 4, dtype=torch.float64)
    z = torch.randn_like(x)
    scales = torch.exp(torch.linspace(-4.6, 0., len(x), dtype=torch.float64))
    batch = x, z, scales
    optimizer_step(field, optimizer, batch, "rectified_linear", 1.0)
    return field, optimizer, batch


@pytest.mark.parametrize("factor", [.25, 1., 4.])
@pytest.mark.parametrize("schedule", ["rectified_linear", "vp_trigonometric"])
def test_all_lambda_function_jacobian_and_original_loss_gradient_transport(factor, schedule):
    field, optimizer, batch = fixture()
    gauged = copy.deepcopy(field)
    other = independent_adam(gauged, optimizer.state_dict())
    apply_condition_gauge(gauged, other, factor)
    x, z, scales = batch
    noisy = x + scales[:, None] * z
    pred, trace = fast_value_and_trace(field, noisy, scales, schedule)
    actual, actual_trace = fast_value_and_trace(gauged, noisy, scales, schedule)
    torch.testing.assert_close(actual, pred, atol=3e-15, rtol=3e-14)
    torch.testing.assert_close(actual_trace, trace, atol=3e-15, rtol=3e-14)
    field.zero_grad(set_to_none=True)
    gauged.zero_grad(set_to_none=True)
    original_loss(field, *batch, schedule).backward()
    original_loss(gauged, *batch, schedule).backward()
    for (name, left), (_, right) in zip(field.named_parameters(), gauged.named_parameters(), strict=True):
        torch.testing.assert_close(right.grad, left.grad / parameter_factor(name, factor), atol=1e-11, rtol=2e-12)


def test_adam_moments_epsilon_steps_and_roundtrip_restore():
    field, optimizer, _ = fixture()
    before = {name: {key: value.clone() if torch.is_tensor(value) else value
                     for key, value in optimizer.state[p].items()} for name, p in field.named_parameters()}
    originals = {name: p.detach().clone() for name, p in field.named_parameters()}
    apply_condition_gauge(field, optimizer, 4.)
    options = {id(p): group for group in optimizer.param_groups for p in group["params"]}
    for name, p in field.named_parameters():
        q = parameter_factor(name, 4.)
        torch.testing.assert_close(p, originals[name] * q, rtol=0, atol=0)
        for key, scale in [("exp_avg", q), ("exp_avg_sq", q*q)]:
            torch.testing.assert_close(optimizer.state[p][key], before[name][key] / scale, rtol=0, atol=0)
        assert torch.equal(optimizer.state[p]["step"], before[name]["step"])
        assert options[id(p)]["eps"] == 1e-6 / q
        assert options[id(p)]["lr"] == .001
    clone = copy.deepcopy(field)
    restored = independent_adam(clone, optimizer.state_dict())
    for (_, a), (_, b) in zip(field.named_parameters(), clone.named_parameters(), strict=True):
        for key in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(optimizer.state[a][key], restored.state[b][key])
            assert optimizer.state[a][key].data_ptr() != restored.state[b][key].data_ptr()


def test_source_coordinate_clipping_has_equal_coefficient_and_transformed_moments():
    field, optimizer, batch = fixture()
    gauged = copy.deepcopy(field)
    other = independent_adam(gauged, optimizer.state_dict())
    factors = apply_condition_gauge(gauged, other, 4.)
    left = optimizer_step(field, optimizer, batch, "rectified_linear", .01)
    right = optimizer_step(gauged, other, batch, "rectified_linear", .01, clip_parameter_factors=factors)
    assert left["unclipped_gradient_norm"] == pytest.approx(right["unclipped_gradient_norm"], rel=2e-13)
    assert left["unclipped_gradient_norm"] > .01
    for (name, a), (_, b) in zip(field.named_parameters(), gauged.named_parameters(), strict=True):
        q = factors[name]
        torch.testing.assert_close(other.state[b]["exp_avg"], optimizer.state[a]["exp_avg"] / q, atol=1e-15, rtol=2e-12)
        torch.testing.assert_close(other.state[b]["exp_avg_sq"], optimizer.state[a]["exp_avg_sq"] / q**2, atol=1e-15, rtol=2e-12)


def test_factor_one_is_exact_noop():
    field, optimizer, _ = fixture()
    model_before = copy.deepcopy(field.state_dict())
    state_before = copy.deepcopy(optimizer.state_dict())
    apply_condition_gauge(field, optimizer, 1.)
    assert all(torch.equal(value, field.state_dict()[name]) for name, value in model_before.items())
    assert optimizer.state_dict()["param_groups"] == state_before["param_groups"]
    for index, state in state_before["state"].items():
        for key, value in state.items():
            assert torch.equal(value, optimizer.state_dict()["state"][index][key])


def test_nontrivial_gauge_continuation_saves_restorable_independent_states(tmp_path):
    config = PilotConfig(ambient_dim=8, intrinsic_dim=2, hidden_dim=12, depth=1,
        batch_size=8, steps=2, validation_count=8, log_interval=2,
        checkpoints=(0, 2), data_law="uniform", device="cpu")
    train_pilot(tmp_path / "source", config)
    source = tmp_path / "source/checkpoint_step_000002.pt"
    result = run(source, tmp_path / "gauge", factor=2., steps=2, device="cpu", interval=2, queries=8)
    assert result["all_optimizer_steps_correct"]
    control = torch.load(tmp_path / "gauge/control/checkpoint_step_000004.pt", weights_only=False)
    gauged = torch.load(tmp_path / "gauge/gauge/checkpoint_step_000004.pt", weights_only=False)
    assert gauged["gauge_metadata"]["cumulative_factor"] == 2.
    assert gauged["gauge_metadata"]["clip_coordinates"] == "source"
    assert torch.equal(control["training_generator_state"], gauged["training_generator_state"])
    from experiments.fm_mechanism_pilot import load_pilot_checkpoint
    field, _ = load_pilot_checkpoint(tmp_path / "gauge/gauge/checkpoint_step_000004.pt", "rectified_linear", "cpu")
    optimizer = independent_adam(field, gauged["optimizers"]["rectified_linear"])
    assert len(optimizer.param_groups) == 3
    assert all(int(state["step"]) == 4 for state in optimizer.state.values())


@pytest.mark.parametrize("law,D,d", [("uniform", 8, 2), ("gaussian", 8, 2), ("circle", 2, 1)])
def test_original_clean_x_continuation_noop_matches_pilot_bitwise(tmp_path, law, D, d):
    config = PilotConfig(ambient_dim=D, intrinsic_dim=d, hidden_dim=12, depth=1,
        batch_size=8, steps=2, validation_count=8, log_interval=2,
        checkpoints=(0, 2), data_law=law, device="cpu")
    train_pilot(tmp_path / "source", config)
    source = tmp_path / "source/checkpoint_step_000002.pt"
    summary = run(source, tmp_path / "gauge", factor=1., steps=2, device="cpu", interval=2, queries=8)
    assert summary["source_unchanged"] and summary["all_optimizer_steps_correct"]
    train_pilot(tmp_path / "source", replace(config, steps=4), resume=source)
    native = torch.load(tmp_path / "source/checkpoint_step_000004.pt", weights_only=False)
    for label in ("control", "gauge"):
        actual = torch.load(tmp_path / f"gauge/{label}/checkpoint_step_000004.pt", weights_only=False)
        for schedule in actual["models"]:
            assert all(torch.equal(value, native["models"][schedule][name]) for name, value in actual["models"][schedule].items())
        assert torch.equal(actual["training_generator_state"], native["training_generator_state"])


@pytest.mark.parametrize("factor", [0., -1., float("nan"), float("inf")])
def test_invalid_factor_is_rejected(factor):
    field, optimizer, _ = fixture()
    with pytest.raises(ValueError, match="positive"):
        apply_condition_gauge(field, optimizer, factor)
