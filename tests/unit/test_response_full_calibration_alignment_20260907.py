import numpy as np
from experiments.response_full_calibration_alignment_20260907 import alignment


def test_orthogonal_error_makes_inverse_gain_suboptimal():
    # u=(1,0), v=(.6,.4): inverse gain restores the u component but amplifies w.
    result = alignment(1., .6**2+.4**2, -.4)
    np.testing.assert_allclose(result['oracle_aligned_gain_beta'], .6)
    np.testing.assert_allclose(result['gamma_from_alignment'], .6/.52)
    assert result['gamma_from_alignment'] < result['inverse_aligned_gain']
    assert result['posterior_oracle_error_at_oracle_ratio'] < result['posterior_oracle_error_if_inverse_gain']


def test_pure_scalar_error_has_exact_inverse_gain_optimum():
    result = alignment(19., .7**2*19., -.3*19.)
    np.testing.assert_allclose(result['gamma_from_alignment'], 1/.7)
    np.testing.assert_allclose(result['unaligned_energy'], 0., atol=3e-15)
    assert result['gamma_alignment_identity_error'] < 3e-15
