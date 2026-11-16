import numpy as np
import pytest
import torch

from experiments.response_full_models_20260907_calibration_selection import fixed_variants, VARIANTS
from experiments.response_full_composition_selection_20260907 import select_scale
from experiments.response_full_multiscale_20260907 import project


def test_all_four_fixed_maps_and_traces_match_actual_autograd_with_negative_gamma():
    torch.manual_seed(652)
    q = torch.randn(5, 4, dtype=torch.float64)
    weight = torch.randn(4, 4, dtype=torch.float64)/3
    offset = torch.randn(4, dtype=torch.float64)
    basis = torch.linalg.qr(torch.randn(4, 2, dtype=torch.float64)).Q
    center = torch.randn(4, dtype=torch.float64)
    predict = lambda x: torch.sin(x@weight.T)+offset
    value = predict(q)
    jac = torch.stack([torch.autograd.functional.jacobian(predict, x) for x in q])
    trace = jac.diagonal(dim1=-2, dim2=-1).sum(-1)
    span_trace = (basis.T@jac@basis).diagonal(dim1=-2, dim2=-1).sum(-1)
    gamma, gp, scale = 1.2, -.25, .0316
    result = fixed_variants(q, value, trace, span_trace, center, basis, scale, gamma, gp)
    functions = dict(original=predict, projected=lambda x: project(predict(x), center, basis),
        calibrated_original=lambda x: x+gamma*(predict(x)-x),
        calibrated_projected=lambda x: project(x, center, basis)+gp*(project(predict(x), center, basis)-project(x, center, basis)))
    for variant in VARIANTS:
        fn = functions[variant]
        actual = fn(q)
        direct = torch.stack([torch.autograd.functional.jacobian(fn, x) for x in q])
        torch.testing.assert_close(result[variant+"_posterior"], actual)
        torch.testing.assert_close(result[variant+"_response"], direct.diagonal(dim1=-2, dim2=-1).sum(-1))
        torch.testing.assert_close(result[variant+"_correction"], ((actual-q)/scale).square().sum(-1))


def test_projection_keeps_original_query_anchor_not_gamma_squared_offspan_error():
    q = torch.tensor([[2., 3.]], dtype=torch.float64)
    b = torch.tensor([[4., 7.]], dtype=torch.float64)
    basis = torch.tensor([[1.], [0.]], dtype=torch.float64)
    result = fixed_variants(q, b, torch.tensor([2.]), torch.tensor([1.]), torch.zeros(2), basis, 1., 0., .5)
    torch.testing.assert_close(result["calibrated_projected_posterior"], torch.tensor([[3., 0.]], dtype=torch.float64))
    torch.testing.assert_close(result["calibrated_projected_correction"], torch.tensor([10.], dtype=torch.float64))
    assert result["calibrated_projected_correction"].item() != .25*result["projected_correction"].item()


def test_validation_selection_uses_only_own_readout_and_preserves_exact_ties():
    scales = np.array([.01, .1, 1.])
    response = np.array([[1., 1.], [1., 1.], [3., 3.]])
    full = np.array([[4., 4.], [2., 2.], [1., 1.]])
    assert select_scale(response, 1, scales)[0] == 0
    assert select_scale(full, 1, scales)[0] == 2
    with pytest.raises(ValueError):
        select_scale(response, 1, scales[::-1])


def test_gamma_one_replays_both_baseline_variants():
    torch.manual_seed(102)
    q, b = [torch.randn(4, 3, dtype=torch.float64) for _ in range(2)]
    basis = torch.eye(3, dtype=torch.float64)[:, :2]
    r, rp = [torch.randn(4, dtype=torch.float64) for _ in range(2)]
    result = fixed_variants(q, b, r, rp, torch.zeros(3), basis, .1, 1., 1.)
    for kind, target in (("calibrated_original", "original"), ("calibrated_projected", "projected")):
        for readout in ("posterior", "response", "correction", "full"):
            torch.testing.assert_close(result[kind+"_"+readout], result[target+"_"+readout])
