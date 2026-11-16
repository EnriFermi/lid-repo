import numpy as np

from experiments.response_full_oracle_20260907 import tube_point
from experiments.response_full_oracle_20260907_gauge import A, gauge_check


def test_rotational_density_gauge_and_exact_adapted_readout():
    x = tube_point(4., .7) + np.array([.03, .01, -.02])
    result = gauge_check(x, .0442778)
    assert np.array_equal(A.T, -A)
    for key in ['adapter_error_residual', 'weighted_divergence_over_density',
                'correction_identity_residual', 'cross_term',
                'rotation_log_mass_residual', 'rotation_posterior_residual']:
        assert abs(result[key]) < 1e-11, (key, result[key])
    assert abs(result['finite_difference_weighted_divergence_over_density']) < 1e-7
    assert result['response_adapted'] == result['response_original']
    assert result['full_adapted'] > result['full_original']


def test_sigma_time_adapter_factor_and_quadratic_amplitude():
    x = tube_point(2., .2)
    small = gauge_check(x, .04, angular_speed=1.)
    large = gauge_check(x, .04, angular_speed=3.)
    first = small['correction_adapted']-small['correction_original']
    second = large['correction_adapted']-large['correction_original']
    np.testing.assert_allclose(second, 9*first, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(first, np.sum(x[1:]**2), atol=1e-12, rtol=1e-12)
