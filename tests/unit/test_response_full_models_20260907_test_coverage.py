import numpy as np
import pytest
import torch

from experiments.response_full_models_20260907_test_coverage import (
    coverage_indices, coverage_metrics, frozen_choices, residual_values,
)


def test_old_ids_keep_order_and_additional_ids_partition_without_replacement():
    old, extra = coverage_indices(np.array([4, 1]), 6)
    np.testing.assert_array_equal(old, [4, 1])
    np.testing.assert_array_equal(extra, [0, 2, 3, 5])
    for invalid in [np.array([1,1]),np.array([6]),np.array([1.])]:
        with pytest.raises(ValueError):coverage_indices(invalid, 6)


def test_frozen_choices_never_optimize_values_and_reject_duplicates():
    first = dict(case="case", passes=1, readout="full", selected_index=2, selected_scale=.1, test_mae=999.)
    second = dict(case="case", readout="full", selected_index=0, selected_scale=.01, eta=-.5, test_mae=1000.)
    result = frozen_choices([first], [second])
    assert result[0]["selected_index"] == 2 and result[1]["selected_index"] == 0
    assert result[1]["eta"] == -.5
    with pytest.raises(ValueError):frozen_choices([first,first], [second])


def test_full_mae_is_weighted_subgroup_mae_not_error_of_mean():
    values = np.array([-1., 3., 1., 2.])
    result = coverage_metrics(values, np.array([1]), 1)
    assert result["old256_mae"] == 2.
    assert result["additional744_mae"] == 1.
    assert result["all1000_mae"] == 1.25
    assert result["all1000_mae"] != abs(values.mean()-1)


def test_fixed_negative_eta_has_actual_blended_trace_and_original_query_correction():
    q = torch.tensor([[.2,-.5]],dtype=torch.float64)
    matrix = torch.tensor([[.8,.2],[.1,.4]],dtype=torch.float64)
    offset = torch.tensor([.1,.3],dtype=torch.float64)
    fn = lambda x:x@matrix.T+offset
    b1, b2 = fn(q), fn(fn(q))
    r1, r2 = torch.trace(matrix).expand(1), torch.trace(matrix@matrix).expand(1)
    eta = -.4
    b, r, c = residual_values(b1,b2,r1,r2,q,.1,eta)
    direct = torch.autograd.functional.jacobian(lambda x:fn(x)+eta*(fn(fn(x))-fn(x)),q[0])
    torch.testing.assert_close(r,torch.trace(direct).expand(1))
    torch.testing.assert_close(c,((b-q)/.1).square().sum(-1))
