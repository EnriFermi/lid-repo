import numpy as np

from experiments.response_full_oracle_20260907_nb import oracle_query_row, spectral_readout
from experiments.response_full_oracle_20260907_spaghetti import harmonic_curve


def test_uncentered_normal_score_svd_is_invariant_to_scalar_gain():
    scores = np.zeros((120, 30))
    scores[:29, :29] = np.eye(29)
    tangent = np.eye(30)[-1]
    first = spectral_readout(scores, tangent)
    second = spectral_readout(.07*scores, tangent)
    assert first['dimension'] == second['dimension'] == 1
    assert first['smallest_vector_tangent_cosine2'] == 1.
    np.testing.assert_allclose(second['singular_values'], .07*first['singular_values'])
    assert first['gap_index'] == 28


def test_normal_score_control_retains_global_null_coordinates():
    query = np.zeros(30)
    query[:20] = harmonic_curve(.7)
    noise = np.random.default_rng(1).normal(size=(3, 30))
    sigma = .01
    post, _, order_error, _ = oracle_query_row((0, query, noise, sigma, np.zeros(30), 1.))
    scores = (post-query[None, :]-sigma*noise)/sigma**2
    np.testing.assert_allclose(scores[:, 20:], -noise[:, 20:]/sigma, atol=1e-13)
    assert order_error < 1e-10
