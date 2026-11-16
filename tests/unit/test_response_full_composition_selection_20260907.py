import numpy as np
import pytest

from experiments.response_full_composition_selection_20260907 import select_scale


def test_selects_validation_mae_and_smallest_exact_tie():
    values = np.array([[0., 2.], [.5, 1.5], [.5, 1.5]])
    index, risk = select_scale(values, 1., [.01, .1, 1.])
    assert index == 1
    np.testing.assert_array_equal(risk, [1., .5, .5])


def test_pointwise_error_not_error_of_mean():
    index, _ = select_scale(np.array([[0., 2.], [1.1, 1.1]]), 1., [.01, .1])
    assert index == 1


def test_invalid_values_and_grid_rejected():
    with pytest.raises(ValueError):
        select_scale(np.array([[np.nan], [1.]]), 1., [.01, .1])
    with pytest.raises(ValueError):
        select_scale(np.ones((2, 1)), 1., [.1, .01])
    with pytest.raises(ValueError):
        select_scale(np.ones((2, 1)), 1., [.1])
