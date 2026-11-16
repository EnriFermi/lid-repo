import pytest

torch = pytest.importorskip("torch")

from experiments.fm_mechanism_norm_probe import feature_dictionary
from experiments.fm_mechanism_pilot import canonical_posterior
from models.neural_fields import NeuralFieldConfig, ScaleConditionedNeuralField


@pytest.mark.parametrize("schedule", ["log_noise", "rectified_linear", "vp_trigonometric"])
def test_normalized_features_reproduce_field_values_and_input_gradients(schedule):
    torch.manual_seed(91)
    field = ScaleConditionedNeuralField(NeuralFieldConfig(
        ambient_dim=8, hidden_dim=16, depth=2, zero_init_output=False,
    )).double().eval()
    x = torch.randn(7, 8, dtype=torch.float64, requires_grad=True)
    lam = torch.logspace(-2, 0, 7, dtype=torch.float64)
    features = feature_dictionary(field, x, lam, schedule)
    actual = canonical_posterior(field, x, lam, schedule)
    reproduced = field.output[-1](features["normalized"])
    torch.testing.assert_close(reproduced, actual, atol=1e-14, rtol=1e-14)
    actual_gradient = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
    probe_gradient = torch.autograd.grad(reproduced.square().sum(), x)[0]
    torch.testing.assert_close(probe_gradient, actual_gradient, atol=1e-13, rtol=1e-13)


def test_variant_definitions_and_equal_feature_dimension():
    torch.manual_seed(92)
    field = ScaleConditionedNeuralField(NeuralFieldConfig(
        ambient_dim=8, hidden_dim=16, depth=2, zero_init_output=False,
    )).double().eval()
    x = torch.randn(7, 8, dtype=torch.float64)
    features = feature_dictionary(field, x, 0.316227766, "rectified_linear")
    h = features["pre_ln_linear"]
    std = ((h-h.mean(-1, keepdim=True)).square().mean(-1, keepdim=True)
           + field.output[0].eps).sqrt()
    torch.testing.assert_close(features["state_std"], std)
    torch.testing.assert_close(features["restore_std"], features["normalized"]*std)
    torch.testing.assert_close(features["remove_final_ln"], field.output[1](h))
    for name in ("normalized", "restore_std", "remove_final_ln", "pre_ln_linear"):
        assert features[name].shape == (7, 16)
