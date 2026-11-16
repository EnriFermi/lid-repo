"""Independent fixed-scale coordinate-transport controls, without training."""

import copy

import pytest

torch = pytest.importorskip("torch")

from experiments.fm_mechanism_coordinate_transport import relative_error, source_gradient
from experiments.fm_mechanism_pilot import canonical_posterior
from models.affine_flow import affine_schedule_state
from models.neural_fields import NeuralFieldConfig, ScaleConditionedNeuralField


@pytest.fixture(autouse=True)
def small_cpu_workload():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(926)
        yield
    torch.set_num_threads(previous_threads)


def make_field():
    return ScaleConditionedNeuralField(NeuralFieldConfig(
        ambient_dim=4, hidden_dim=7, depth=2, condition_dim=5,
        fourier_features=3, zero_init_output=False,
    )).double().eval()


def transport(field, scale, source, destination):
    lam = torch.tensor(scale, dtype=torch.float64)
    factor = float(affine_schedule_state(lam, destination).alpha /
                   affine_schedule_state(lam, source).alpha)
    changed = copy.deepcopy(field)
    with torch.no_grad():
        changed.input.weight.div_(factor)
    return changed, factor


@pytest.mark.parametrize("source,destination", [
    ("rectified_linear", "vp_trigonometric"),
    ("vp_trigonometric", "rectified_linear"),
])
@pytest.mark.parametrize("scale", [.03, .316227766, .8])
def test_transport_preserves_values_full_jacobian_and_full_readout(source, destination, scale):
    field = make_field()
    old_state = copy.deepcopy(field.state_dict())
    changed, _ = transport(field, scale, source, destination)
    point = torch.randn(4, dtype=torch.float64)

    def evaluate(model, schedule):
        function = lambda x: canonical_posterior(model, x[None], scale, schedule)[0]
        value = function(point)
        jacobian = torch.autograd.functional.jacobian(function, point)
        full = jacobian.trace() + (value - point).square().sum() / scale**2
        return value, jacobian, full

    for actual, expected in zip(evaluate(changed, destination), evaluate(field, source)):
        torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    for name, value in field.state_dict().items():
        assert torch.equal(value, old_state[name])
        if name != "input.weight":
            assert torch.equal(changed.state_dict()[name], value)


@pytest.mark.parametrize("source,destination", [
    ("rectified_linear", "vp_trigonometric"),
    ("vp_trigonometric", "rectified_linear"),
])
def test_raw_gradient_transport_and_common_denominator(source, destination):
    field = make_field()
    scale = .316227766
    changed, factor = transport(field, scale, source, destination)
    points = torch.randn(259, 4, dtype=torch.float64)  # Crosses source_gradient's chunk boundary.
    targets = torch.randn_like(points)
    mask = points[:, 0] > 0
    losses, gradients = [], []
    for selected in (mask, ~mask, torch.ones_like(mask)):
        loss, grads = source_gradient(field, points, targets, selected, scale, source)
        new_loss, new_grads = source_gradient(changed, points, targets, selected, scale, destination)
        expected = [g * factor if name == "input.weight" else g
                    for (name, _), g in zip(field.named_parameters(), grads)]
        assert new_loss == pytest.approx(loss, rel=2e-13, abs=2e-13)
        assert relative_error(new_grads, expected) < 2e-12
        losses.append(loss)
        gradients.append(grads)
    assert losses[0] + losses[1] == pytest.approx(losses[2], rel=2e-13)
    for left, right, total in zip(*gradients):
        torch.testing.assert_close(left + right, total, rtol=2e-12, atol=2e-12)


def test_one_weight_rescaling_is_not_a_cross_scale_equivalence():
    field = make_field()
    changed, _ = transport(field, .316227766, "rectified_linear", "vp_trigonometric")
    points = torch.randn(5, 4, dtype=torch.float64)
    source = canonical_posterior(field, points, .03, "rectified_linear")
    destination = canonical_posterior(changed, points, .03, "vp_trigonometric")
    assert float((source - destination).abs().max()) > 1e-5


def test_parameter_metric_must_also_transform_to_preserve_a_descent_step():
    field = make_field()
    scale = .316227766
    changed, factor = transport(field, scale, "rectified_linear", "vp_trigonometric")
    naive = copy.deepcopy(changed)
    points = torch.randn(9, 4, dtype=torch.float64)
    target = torch.randn_like(points)
    mask = torch.ones(len(points), dtype=torch.bool)
    _, grads = source_gradient(field, points, target, mask, scale, "rectified_linear")
    _, new_grads = source_gradient(changed, points, target, mask, scale, "vp_trigonometric")
    with torch.no_grad():
        for (name, old), transformed, untouched_metric, g, new_g in zip(
            field.named_parameters(), changed.parameters(), naive.parameters(), grads, new_grads
        ):
            old.add_(g, alpha=-1e-4)
            multiplier = 1 / factor**2 if name == "input.weight" else 1.0
            transformed.add_(new_g, alpha=-1e-4 * multiplier)
            untouched_metric.add_(new_g, alpha=-1e-4)
    expected = canonical_posterior(field, points, scale, "rectified_linear")
    actual = canonical_posterior(changed, points, scale, "vp_trigonometric")
    naive_value = canonical_posterior(naive, points, scale, "vp_trigonometric")
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    assert float((naive_value - expected).abs().max()) > 1e-7
