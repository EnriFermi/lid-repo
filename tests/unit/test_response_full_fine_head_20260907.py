import pytest
import torch

from experiments.response_full_fine_head_20260907 import fine_mask, fit_fine_head, objective_partitions
from experiments.response_full_models_20260907_convex_head import (
    body_hash, capture_features, replaced_head, residual_targets,
)
from experiments.response_full_models_20260907_rb_training import cached_edm_loss
from tests.unit.test_response_full_models_20260907_rb_training import tiny_field


def test_fine_mask_is_fixed_physical_threshold_and_retains_boundary():
    scale = torch.tensor([.01, .03, .0300000001, 1.], dtype=torch.float64)
    assert fine_mask(scale).tolist() == [True, True, False, False]
    with pytest.raises(ValueError):
        fine_mask(torch.tensor([0., .02, 1.]))
    with pytest.raises(ValueError):
        fine_mask(torch.tensor([.01, .02]))


def test_single_fine_fit_ignores_complement_targets_and_uses_same_svd_rule():
    torch.manual_seed(707)
    features = torch.randn(40, 5, dtype=torch.float64)
    design = torch.cat([features, torch.ones(40, 1, dtype=torch.float64)], 1)
    target = torch.randn(40, 3, dtype=torch.float64)
    sigma = torch.cat([torch.full((25,), .02), torch.full((15,), .1)]).double()
    coef, mask, spectrum, residual, receipt = fit_fine_head(design, target, sigma)
    changed = target.clone()
    changed[~mask] += 1e5
    other = fit_fine_head(design, changed, sigma)[0]
    torch.testing.assert_close(coef, other, rtol=0, atol=0)
    assert receipt["rcond"] == torch.finfo(torch.float64).eps * 25
    assert receipt["rank"] == 6 and receipt["fine_rows"] == 25
    assert receipt["normalized_stationarity"] < 1e-13
    torch.testing.assert_close(residual, design[mask] @ coef-target[mask])


def test_partition_objective_recombines_and_equals_actual_edm_loss():
    torch.manual_seed(708)
    field = tiny_field().eval()
    observed = torch.randn(24, 4, dtype=torch.float64)
    sigma = torch.cat([torch.full((15,), .02), torch.full((9,), .2)]).double()
    target = torch.randn_like(observed)
    features = capture_features(field, observed, sigma)
    design = torch.cat([features, torch.ones(24, 1, dtype=torch.float64)], 1)
    transformed = residual_targets(observed, sigma, target)
    coefficient, mask, _, _, _ = fit_fine_head(design, transformed, sigma)
    fitted = replaced_head(field, coefficient)
    assert body_hash(fitted) == body_hash(field)
    parts = objective_partitions(design, transformed, coefficient, mask)
    expected = (15*parts["fine"]["mse"] + 9*parts["rest"]["mse"])/24
    assert abs(expected-parts["all"]["mse"]) < 1e-9
    for key, subset in {"fine": mask, "rest": ~mask, "all": torch.ones_like(mask)}.items():
        actual = cached_edm_loss(fitted, observed[subset], sigma[subset], target[subset])
        torch.testing.assert_close(actual, actual.new_tensor(parts[key]["mse"]), rtol=1e-10, atol=1e-10)


def test_fit_rejects_hidden_nonconstant_intercept_column():
    design = torch.randn(12, 4, dtype=torch.float64)
    target = torch.randn(12, 2, dtype=torch.float64)
    sigma = torch.linspace(.01, .1, 12)
    with pytest.raises(ValueError, match="intercept"):
        fit_fine_head(design, target, sigma)
