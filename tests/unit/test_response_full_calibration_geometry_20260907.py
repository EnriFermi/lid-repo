import numpy as np
from experiments.response_full_calibration_geometry_20260907 import regression_components, calibrated_local_trace


def test_normal_undercontraction_has_inverse_gain_calibration():
    rng = np.random.default_rng(73)
    u = rng.normal(size=(5, 40, 4)); u[..., 0] = 0
    tangent = np.zeros((5, 4)); tangent[:, 0] = 1
    result = regression_components(.7*u, u, tangent)
    np.testing.assert_allclose(result['oracle_ratio'], 1/.7, atol=2e-15)
    assert result['normal_model_energy_fraction'] == 1.
    np.testing.assert_allclose(calibrated_local_trace(3*.3, result['oracle_ratio'], 3), 0., atol=2e-15)


def test_finite_tangent_component_and_error_identity_are_retained():
    rng = np.random.default_rng(37)
    u = rng.normal(size=(4, 80, 6)); v = .8*u+.2*rng.normal(size=u.shape)
    tangent = np.zeros((4, 6)); tangent[:, 0] = 1
    result = regression_components(v, u, tangent)
    assert 0 < result['normal_oracle_energy_fraction'] < 1
    assert result['weighted_ratio_identity_error'] < 2e-15
    assert result['error_ratio_identity_error'] < 2e-15


def test_trace_identity_does_not_force_oracle_clean_response():
    original_t, original_n = np.array([.98, 1.03]), np.array([.7, -.2])
    gamma = 1.1
    total = calibrated_local_trace(original_t, gamma, 1)+calibrated_local_trace(original_n, gamma, 19)
    np.testing.assert_allclose(total, calibrated_local_trace(original_t+original_n, gamma, 20))
    assert not np.allclose(total, 1.)
