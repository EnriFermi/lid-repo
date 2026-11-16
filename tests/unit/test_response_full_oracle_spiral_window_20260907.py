import numpy as np
from experiments.response_full_oracle_spiral_window_20260907 import exact_latents, prepare, SCALES
from experiments.response_full_oracle_20260907 import spiral_curve, spiral_posterior


def test_exact_latents_match_original_continuous_sampler_test_offset():
    ids = np.array([1, 87, 999])
    expected = np.random.RandomState(0).uniform(1, 100, 102000)[101000:][ids]
    np.testing.assert_array_equal(exact_latents(ids), expected)


def test_prepare_does_not_invert_float32_coordinates():
    ids = np.array([0, 2, 3])
    source = {'query_ids': ids, 'normalization_mean': np.zeros(30),
              'normalization_rms': .02, 'queries': np.zeros((3, 30), dtype=np.float32)}
    result = prepare(source)
    np.testing.assert_array_equal(result['query_raw'][:, :2], spiral_curve(exact_latents(ids)))
    np.testing.assert_array_equal(result['queries'], result['query_raw']/.02)
    np.testing.assert_array_equal(result['scales'], SCALES)


def test_actual_spiral_resolves_at_sufficiently_fine_scale():
    q = spiral_curve(80.)
    result = spiral_posterior(q, 2e-7, order=96, cutoff=12.)
    check = spiral_posterior(q, 2e-7, order=48, cutoff=14.)
    assert abs(result.full-1) < 1e-5
    assert result.metadata['other_turn_mass'] < 1e-6
    # Large absolute phase and a tiny angular window have finite argument-
    # reduction precision; this is still far below the declared 1e-6 gate.
    assert abs(result.full-check.full) < 1e-7
