import pytest

torch = pytest.importorskip("torch")

from experiments.fm_mechanism_head_refit import final_features, solve_weighted_head, replaced_head
from experiments.fm_mechanism_pilot import canonical_posterior
from models.neural_fields import NeuralFieldConfig, ScaleConditionedNeuralField


def test_weighted_head_recovers_known_affine_map() -> None:
    torch.manual_seed(19)
    features = torch.randn(500, 12, dtype=torch.float64)
    coefficient = torch.randn(13, 3, dtype=torch.float64)
    target = torch.cat([features, torch.ones(500, 1, dtype=torch.float64)], 1) @ coefficient
    lam = torch.logspace(-2, 0, 500, dtype=torch.float64)
    fit, audit = solve_weighted_head(features, target, lam)
    torch.testing.assert_close(fit, coefficient, atol=1e-11, rtol=1e-11)
    assert audit["rank"] == 13


def test_captured_features_are_actual_final_linear_inputs() -> None:
    torch.manual_seed(17)
    field = ScaleConditionedNeuralField(NeuralFieldConfig(ambient_dim=8, hidden_dim=16, depth=2, zero_init_output=False)).double()
    x = torch.randn(9, 8, dtype=torch.float64)
    lam = torch.logspace(-2, 0, 9, dtype=torch.float64)
    features = final_features(field, x, lam, "rectified_linear", chunk=4)
    with torch.no_grad():
        actual = canonical_posterior(field, x, lam, "rectified_linear")
    torch.testing.assert_close(field.output[-1](features), actual, atol=1e-14, rtol=1e-14)


def test_replaced_head_keeps_body_and_tangent_support() -> None:
    torch.manual_seed(17)
    field = ScaleConditionedNeuralField(NeuralFieldConfig(ambient_dim=8, hidden_dim=16, depth=2)).double()
    basis = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64)).Q
    coefficient = torch.randn(17, 2, dtype=torch.float64)
    refit = replaced_head(field, coefficient, basis[:, :2])
    for key, value in field.state_dict().items():
        if key not in {"output.2.weight", "output.2.bias"}:
            torch.testing.assert_close(value, refit.state_dict()[key], atol=0, rtol=0)
    torch.testing.assert_close(basis[:, 2:].T @ refit.output[-1].weight, torch.zeros(6, 16, dtype=torch.float64), atol=1e-7, rtol=0)


def test_native_output_weighting_matches_canonical_physical_risk() -> None:
    torch.manual_seed(19)
    features = torch.randn(500, 12, dtype=torch.float64)
    target = torch.randn(500, 3, dtype=torch.float64)
    lam = torch.logspace(-2, 0, 500, dtype=torch.float64)
    alpha = 1 / (1 + lam)
    fit, _ = solve_weighted_head(features, alpha[:, None] * target, alpha * lam)
    augmented = torch.cat([features, torch.ones(500, 1, dtype=torch.float64)], 1)
    direct = torch.linalg.lstsq(augmented / (alpha * lam)[:, None], target / lam[:, None], driver="gelsd", rcond=1e-10).solution
    torch.testing.assert_close(fit, direct, atol=1e-12, rtol=1e-12)
    physical = (augmented @ fit / alpha[:, None] - target) / lam[:, None]
    native = (augmented @ fit - alpha[:, None] * target) / (alpha * lam)[:, None]
    torch.testing.assert_close(physical, native, atol=1e-13, rtol=1e-13)
