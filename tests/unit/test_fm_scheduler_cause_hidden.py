import pytest
import torch

from experiments.fm_scheduler_cause_hidden import hidden_parts, energy_accounting
from experiments.fm_scheduler_cause_radial import pre_final_state
from models.neural_fields import NeuralFieldConfig, ScaleConditionedNeuralField


def model():
    torch.manual_seed(942)
    return ScaleConditionedNeuralField(NeuralFieldConfig(
        ambient_dim=4, hidden_dim=12, depth=3, condition_dim=8, fourier_features=4,
        condition_transform="linear", zero_init_output=False)).double().eval()


@pytest.mark.parametrize("schedule", ["rectified_linear", "vp_trigonometric"])
def test_exact_hidden_accounting(schedule):
    field = model()
    points = torch.randn(8, 4, dtype=torch.float64)
    parts = hidden_parts(field, points, .316, schedule)
    torch.testing.assert_close(parts["state"], pre_final_state(field, points, .316, schedule), atol=1e-15, rtol=1e-15)
    torch.testing.assert_close(parts["state"], parts["linear"]+parts["input_constant"]+parts["residual_updates"], atol=1e-15, rtol=1e-15)
    result = energy_accounting(parts)
    assert result["signed_projection_sum"] == pytest.approx(1., abs=1e-14)
    assert sum(result[f"{name}_energy_ratio"] for name in ("linear", "input_constant", "residual_updates"))+result["twice_all_cross_ratio"] == pytest.approx(1., abs=1e-14)


def test_fixed_condition_preserves_linear_component_and_changes_conditioned_parts():
    field = model()
    points = torch.randn(8, 4, dtype=torch.float64)
    actual = hidden_parts(field, points, .3, "rectified_linear")
    frozen = hidden_parts(field, points, .3, "rectified_linear", condition_scale=.01)
    torch.testing.assert_close(actual["linear"], frozen["linear"], atol=0, rtol=0)
    assert not torch.allclose(actual["input_constant"], frozen["input_constant"])
    assert not torch.allclose(actual["residual_updates"], frozen["residual_updates"])


def test_leading_asymptotic_state_and_no_ln_output():
    field = model()
    field.output[0] = torch.nn.Identity()
    direction = torch.randn(3, 4, dtype=torch.float64)
    scale, schedule = .3, "rectified_linear"
    one = hidden_parts(field, direction, scale, schedule)
    huge = hidden_parts(field, direction*1e7, scale, schedule)
    torch.testing.assert_close(huge["state"]/1e7, one["linear"], atol=1e-6, rtol=0)
    expected = torch.nn.functional.linear(torch.relu(one["linear"]), field.output[-1].weight)
    torch.testing.assert_close(field.output(huge["state"])/1e7, expected, atol=1e-6, rtol=0)
