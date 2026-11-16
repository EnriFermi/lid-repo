import numpy as np
import pytest

from experiments.response_full_richardson_20260907 import extrapolate, halfline_readouts, analyze_pairwise


def test_quadratic_term_cancels_without_dimension_knowledge():
    scales = np.logspace(-2, 0, 9)
    d = np.array([1., 2., 5.])
    values = d+scales[:, None]**2*np.array([.3, -9., 2.])
    np.testing.assert_allclose(extrapolate(values, scales), np.broadcast_to(d, (8, 3)), atol=3e-15)


def test_quartic_remainder_and_inverse_square_error_have_predicted_sign():
    scales = np.logspace(-2, 0, 9)
    ratio = (scales[1:]/scales[:-1])**2
    np.testing.assert_allclose(extrapolate(scales**4, scales), -ratio*scales[:-1]**4, rtol=1e-13)
    np.testing.assert_allclose(extrapolate(scales**-2, scales), (1+1/ratio)*scales[:-1]**-2, rtol=1e-13)


def test_flat_boundary_full_can_worsen_and_far_interior_can_overshoot():
    ratio = 10**.25
    r, f = halfline_readouts(np.array([.5, .5/ratio]))
    assert extrapolate(f, [1, ratio])[0] < f[0] < 1
    assert r[0] < extrapolate(r, [1, ratio])[0] < 1
    r, f = halfline_readouts(np.array([4, 4/ratio]))
    assert extrapolate(f, [1, ratio])[0] > 1.03
    assert 1-f[0] < .001
    r, f = halfline_readouts(np.zeros(2))
    np.testing.assert_allclose(extrapolate(r, [1, ratio]), 1-2/np.pi)
    np.testing.assert_allclose(extrapolate(f, [1, ratio]), 1.)


def test_oracle_alignment_rejects_different_queries():
    data = {'scales': np.array([.1, .2]), 'query_ids': np.arange(2), 'queries': np.eye(2)}
    r = np.ones((2, 2))
    oracle = {**data, 'response': r, 'full': r, 'queries': np.zeros((2, 2))}
    with pytest.raises(AssertionError):
        analyze_pairwise('case', 'spaghetti', data, r, r, oracle)


def test_scale_validation_and_pairwise_target_operator_are_separate():
    for scales in [[.1, .1], [.2, .1], [0, .1]]:
        with pytest.raises(ValueError):
            extrapolate(np.ones((2, 3)), scales)
    scales = np.array([.1, .2])
    data = {'scales': scales, 'query_ids': np.arange(2), 'queries': np.eye(2)}
    r = 1+scales[:, None]**2*np.ones((2, 2))
    oracle = {**data, 'response': r, 'full': r}
    arrays, rows = analyze_pairwise('case', 'spaghetti', data, r, r, oracle)
    assert rows[0]['extrapolated_same_scale_oracle_mae'] > 0
    assert rows[0]['extrapolated_extrapolated_oracle_operator_mae'] == 0
    assert arrays['fine_scales'][0] == .1
