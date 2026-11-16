import pytest
import torch

from experiments.fm_mechanism_pilot import canonical_posterior
from experiments.fm_scheduler_cause_radial import pre_final_state, radial_chain, odd_components
from models.neural_fields import NeuralFieldConfig, ScaleConditionedNeuralField


def model():
    torch.manual_seed(826)
    return ScaleConditionedNeuralField(NeuralFieldConfig(
        ambient_dim=4, hidden_dim=12, depth=2, condition_dim=8, fourier_features=4,
        condition_transform="linear", zero_init_output=False)).double().eval()


@pytest.mark.parametrize("schedule", ["rectified_linear", "vp_trigonometric"])
@pytest.mark.parametrize("normalization", ["layernorm", "none"])
def test_exact_final_normalizer_chain_and_original_forward(schedule, normalization):
    field = model()
    if normalization == "none":
        field.output[0] = torch.nn.Identity()
    points = torch.randn(5, 4, dtype=torch.float64)
    values = radial_chain(field, points, .316227766, schedule)
    expected, jvp = torch.autograd.functional.jvp(
        lambda x: canonical_posterior(field, x, .316227766, schedule), points, points)
    torch.testing.assert_close(values["value"], expected, atol=1e-14, rtol=1e-13)
    torch.testing.assert_close(values["numerator"] + values["denominator"], jvp, atol=1e-14, rtol=1e-12)
    torch.testing.assert_close(values["numerator_euler_centered"] + values["denominator"], values["euler_centered"], atol=1e-14, rtol=1e-12)
    if normalization == "none":
        assert torch.equal(values["denominator"], torch.zeros_like(values["denominator"]))


def test_odd_euler_uses_the_correct_sign_for_negative_ray_jvp():
    field = model()
    x = torch.randn(3, 4, dtype=torch.float64)
    scale, schedule = .316227766, "rectified_linear"
    values = odd_components(radial_chain(field, x, scale, schedule), radial_chain(field, -x, scale, schedule))
    def odd(y):
        return (canonical_posterior(field, y, scale, schedule) - canonical_posterior(field, -y, scale, schedule)) / 2
    prediction, derivative = torch.autograd.functional.jvp(odd, x, x)
    torch.testing.assert_close(values["euler_centered"], derivative - prediction, atol=1e-14, rtol=1e-12)


def test_frozen_denominator_local_function_has_same_value_and_numerator_derivative():
    field = model()
    x = torch.randn(3, 4, dtype=torch.float64)
    scale, schedule = .316227766, "rectified_linear"
    result = radial_chain(field, x, scale, schedule)
    fixed_std = result["pre_final_std"][:, None]
    def counterfactual(y):
        h = pre_final_state(field, y, scale, schedule)
        z = h - h.mean(-1, keepdim=True)
        normalized = field.output[0].weight * z / fixed_std + field.output[0].bias
        return field.output[-1](field.output[1](normalized))
    value, jvp = torch.autograd.functional.jvp(counterfactual, x, x)
    torch.testing.assert_close(value, result["value"], atol=1e-14, rtol=1e-12)
    torch.testing.assert_close(jvp, result["numerator"], atol=1e-14, rtol=1e-12)


def test_origin_correction_removes_constant_output_from_euler():
    field = model()
    with torch.no_grad():
        field.output[-1].weight.zero_()
        field.output[-1].bias.fill_(2.)
    result = radial_chain(field, torch.randn(3, 4, dtype=torch.float64), .3, "rectified_linear")
    torch.testing.assert_close(result["euler_centered"], torch.zeros_like(result["value"]), atol=0, rtol=0)
    assert torch.equal(result["euler_raw"], torch.full_like(result["value"], -2.))
