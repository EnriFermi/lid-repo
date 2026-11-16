import numpy as np

from experiments.response_full_local_geometry_20260907 import exp_coefficients, local_coefficients
from experiments.response_full_oracle_20260907 import tube_point, tube_posterior


def test_uniform_sphere_and_cylinder_have_opposite_readout_order():
    sphere = local_coefficients(ii_squared=2., h_squared=4.)
    cylinder = local_coefficients(ii_squared=1., h_squared=1.)
    assert sphere['response_coefficient'] == -1.
    assert sphere['full_coefficient'] == 0.
    assert sphere['twice_response_plus_correction'] < 0.
    assert cylinder['response_coefficient'] == 0.
    assert cylinder['full_coefficient'] == .25
    assert cylinder['twice_response_plus_correction'] > 0.


def test_original_exp_response_cancellation_and_strict_comparison_sign():
    for u in np.linspace(.1, 7.9, 20):
        result = exp_coefficients(u)
        t = 9*np.exp(-2*u); g = 1+t
        np.testing.assert_allclose(result['response_coefficient'], -3*t/g**3, rtol=1e-6, atol=1e-10)
        expected = (4*(t-.25)**2+.75+1/(4*t))/g**3
        np.testing.assert_allclose(result['twice_response_plus_correction'], expected, rtol=1e-11)
        assert result['twice_response_plus_correction'] > 0.
        uniform_area = exp_coefficients(u, 'uniform_area')
        np.testing.assert_allclose(uniform_area['response_coefficient'], 1/g**2, rtol=1e-9)


def test_population_oracle_coefficients_converge_under_scale_halving():
    for prior in ['uniform_u', 'uniform_area']:
        target = exp_coefficients(1., prior)
        errors = []
        for sigma in [.01, .005]:
            result = tube_posterior(tube_point(1., .7), sigma, order=256, cutoff=14., prior=prior)
            measured = np.array([(result.response-2)/sigma**2, result.correction/sigma**2])
            expected = np.array([target['response_coefficient'], target['correction_coefficient']])
            errors.append(np.abs(measured-expected))
        assert np.max(errors[-1]) < 2e-5
        assert np.all(np.asarray(errors[0])/np.asarray(errors[1]) > 3.9)
        assert np.all(np.asarray(errors[0])/np.asarray(errors[1]) < 4.1)
