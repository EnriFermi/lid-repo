import numpy as np

from experiments.response_full_models_20260907_oracle_audit import decompose


def test_exact_displacement_identity_keeps_signed_cross_and_derivative_error():
    rng = np.random.default_rng(901)
    q = rng.normal(size=(7, 5))
    oracle = rng.normal(size=(3, 7, 5))
    error = -.3*(oracle-q[None])
    r = rng.normal(size=(3, 7))
    rstar = rng.normal(size=(3, 7))
    result = decompose(q, oracle+error, oracle, r, rstar, [.01, .1, 1.])
    assert np.all(result["oracle_cross"] <= 0)
    np.testing.assert_allclose(result["correction"], .7**2*result["oracle_correction"])
    np.testing.assert_allclose(result["full"]-result["oracle_full"],
        result["response_error"]+result["oracle_cross"]+result["posterior_error_energy"])


def test_oracle_exactly_recovered_has_zero_error_without_assuming_zero_correction():
    q = np.ones((4, 3))
    oracle = np.full((2, 4, 3), .5)
    response = np.full((2, 4), 2.)
    result = decompose(q, oracle, oracle, response, response, [.1, 1.])
    assert np.all(result["oracle_correction"] > 0)
    for key in ["response_error", "posterior_error_energy", "oracle_cross"]:
        np.testing.assert_array_equal(result[key], np.zeros((2, 4)))
