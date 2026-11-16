import numpy as np
from experiments.response_full_nb_20260907 import analyze_scores,estimate_from_spectrum


def test_selector_uses_absolute_not_logarithmic_gap():
    assert estimate_from_spectrum(np.array([100.,60.,59.,20.]))==3


def test_exact_normal_subspace_detects_missing_tangent():
    rng=np.random.default_rng(1)
    score=rng.standard_normal((2,200,4));score[:,:,0]=0
    tangent=np.tile([1.,0.,0.,0.],(2,1))
    result=analyze_scores(score,tangent)
    np.testing.assert_array_equal(result["dimension_estimate"],[1,1])
    np.testing.assert_allclose(result["smallest_vector_tangent_cosine"],1.,atol=1e-12)


def test_rotation_preserves_estimate_and_tangent_alignment():
    rng=np.random.default_rng(2)
    score=rng.standard_normal((3,200,4));score[:,:,0]*=.01
    tangent=np.tile([1.,0.,0.,0.],(3,1))
    rotation=np.linalg.qr(rng.standard_normal((4,4)))[0]
    original=analyze_scores(score,tangent)
    rotated=analyze_scores(score@rotation,tangent@rotation)
    np.testing.assert_array_equal(original["dimension_estimate"],rotated["dimension_estimate"])
    np.testing.assert_allclose(original["singular_values"],rotated["singular_values"],atol=1e-12)
    np.testing.assert_allclose(original["smallest_vector_tangent_cosine"],rotated["smallest_vector_tangent_cosine"],atol=1e-12)
