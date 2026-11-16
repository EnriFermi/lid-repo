import pytest

torch = pytest.importorskip("torch")

from experiments.fm_mechanism_fast_trace import fast_value_and_trace
from experiments.fm_mechanism_pilot import canonical_posterior
from models.neural_fields import NeuralFieldConfig, ScaleConditionedNeuralField


def actual_field(dimension=8, hidden=13, final_norm=True):
    torch.manual_seed(73)
    field = ScaleConditionedNeuralField(NeuralFieldConfig(
        ambient_dim=dimension, hidden_dim=hidden, depth=2,
        zero_init_output=False, dropout=.2,
    )).double().eval()
    if not final_norm:
        field.output[0] = torch.nn.Identity()
    else:
        with torch.no_grad():
            field.output[0].weight.uniform_(.3, 1.6)
            field.output[0].bias.uniform_(-.4, .4)
    return field


def autograd_trace(field, inputs, scales, schedule, **kwargs):
    value = canonical_posterior(field, inputs, scales, schedule, **kwargs)
    trace = inputs.new_zeros(len(inputs))
    for index in range(inputs.shape[-1]):
        gradient = torch.autograd.grad(value[:, index].sum(), inputs, create_graph=True, retain_graph=True)[0]
        trace = trace+gradient[:, index]
    return value, trace


@pytest.mark.parametrize("schedule", ["log_noise", "rectified_linear", "vp_trigonometric"])
@pytest.mark.parametrize("final_norm", [True, False])
@pytest.mark.parametrize("output_coordinate", ["data", "native"])
def test_exact_value_and_trace_match_actual_autograd(schedule, final_norm, output_coordinate):
    field = actual_field(final_norm=final_norm)
    inputs = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
    scales = torch.tensor([.01, .1, .316, 1.], dtype=torch.float64)
    kwargs = {"output_coordinate": output_coordinate}
    expected_value, expected_trace = autograd_trace(field, inputs, scales, schedule, **kwargs)
    value, trace = fast_value_and_trace(field, inputs, scales, schedule, **kwargs)
    torch.testing.assert_close(value, expected_value, atol=2e-14, rtol=2e-13)
    torch.testing.assert_close(trace, expected_trace, atol=2e-13, rtol=2e-12)


def test_trace_and_full_gradients_are_preserved():
    field = actual_field(dimension=5, hidden=9)
    inputs = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    scale = .316227766
    value, trace = fast_value_and_trace(field, inputs, scale, "rectified_linear")
    expected_value, expected_trace = autograd_trace(field, inputs, scale, "rectified_linear")
    loss = (trace+(value-inputs).square().sum(-1)/scale**2).square().mean()
    expected_loss = (expected_trace+(expected_value-inputs).square().sum(-1)/scale**2).square().mean()
    variables = (inputs, field.input.weight, field.output[0].weight,
                 field.output[-1].weight, field.condition_embedding.projection[0].weight)
    gradients = torch.autograd.grad(loss, variables, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected_loss, variables)
    for actual, expected in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, atol=5e-10, rtol=5e-10)


def test_channel_input_native_output_and_active_dropout_gate():
    field = actual_field()
    inputs = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
    kwargs = {"input_coordinate": "channel", "output_coordinate": "native"}
    value, trace = fast_value_and_trace(field, inputs, .2, "rectified_linear", **kwargs)
    expected_value, expected_trace = autograd_trace(field, inputs, .2, "rectified_linear", **kwargs)
    torch.testing.assert_close(value, expected_value)
    torch.testing.assert_close(trace, expected_trace)
    field.train()
    with pytest.raises(ValueError, match="dropout"):
        fast_value_and_trace(field, inputs, .2, "rectified_linear")
