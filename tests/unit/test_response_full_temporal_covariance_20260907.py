import numpy as np
from experiments.response_full_temporal_covariance_20260907 import temporal_covariance


def test_within_query_centering_is_not_spatial_pooling():
    t = np.array([-.5, .5])[:, None]
    offsets = np.array([10., 100.])[None, :]
    r, c = offsets+t, offsets-t
    summary, _ = temporal_covariance(r, c, 1)
    assert np.mean((r-r.mean())*(c-c.mean())) > 0
    assert summary['cov_response_correction'] == -.25
    assert summary['var_full'] == 0
    assert summary['full_less_variable_query_fraction'] == 1


def test_negative_covariance_does_not_suffice_for_stabilization():
    t = np.array([-1., 1.])[:, None]
    summary, _ = temporal_covariance(t, 5-3*t, 1)
    assert summary['cov_response_correction'] == -3
    assert summary['var_full'] == 4
    assert summary['var_response'] == 1
    assert summary['full_less_variable_query_fraction'] == 0


def test_full_can_be_stable_but_less_accurate():
    r = np.array([[.9], [1.1]])
    c = 4-r
    summary, _ = temporal_covariance(r, c, 1)
    assert summary['var_full'] == 0
    assert summary['mean_snapshot_full_dimension_mae'] > summary['mean_snapshot_response_dimension_mae']


def test_variance_and_mse_identities_for_general_arrays():
    rng = np.random.default_rng(123)
    r, c = rng.normal(size=(16, 64)), rng.uniform(size=(16, 64))
    summary, _ = temporal_covariance(r, c, 2)
    assert summary['variance_identity_max_error'] < 1e-13
    assert summary['response_mse_identity_error'] < 1e-13
    assert summary['full_mse_identity_error'] < 1e-13
