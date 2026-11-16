import numpy as np
from experiments.response_full_calibration_pointwise_20260907 import pointwise_alignment


def test_variable_positive_gain_is_not_direction_error():
    u = np.array([[1., 0.], [0., 2.], [1., 1.]])
    v = np.array([.5, 1., 2.])[:, None]*u
    row, saved = pointwise_alignment(v, u)
    assert row['global_unaligned_energy'] > 0
    assert row['pointwise_direction_energy'] < 1e-28
    assert row['pointwise_oracle_gamma_value_error_per_coordinate'] < 1e-28
    np.testing.assert_allclose(saved['gamma_pointwise_oracle'], [2., 1., .5])


def test_equal_gain_wrong_directions_cannot_be_repaired_by_scalars():
    u = np.array([[1., 0.], [1., 0.]])
    v = np.array([[1., 1.], [1., -1.]])
    row, _ = pointwise_alignment(v, u)
    assert row['varying_gain_energy'] == 0
    assert row['pointwise_direction_energy'] == 1
    assert row['pointwise_oracle_gamma_value_error_per_coordinate'] == .25


def test_zero_vectors_and_negative_scalar_are_defined():
    u = np.array([[0., 0.], [1., 0.], [1., 0.]])
    v = np.array([[0., 1.], [0., 0.], [-2., 0.]])
    row, saved = pointwise_alignment(v, u)
    assert row['zero_oracle_vector_count'] == 1
    assert row['zero_model_vector_count'] == 1
    np.testing.assert_allclose(saved['gamma_pointwise_oracle'], [0., 0., -.5])
    np.testing.assert_allclose(saved['pointwise_oracle_value_error'], [0., 1., 0.])


def test_split_and_lower_bound_for_general_vectors():
    rng = np.random.default_rng(42)
    u, v = rng.normal(size=(2, 3, 7)), rng.normal(size=(2, 3, 7))
    row, saved = pointwise_alignment(v, u, 1.3)
    assert row['split_identity_max_error'] < 1e-13
    assert row['pointwise_lower_bound_identity_max_error'] < 1e-13
    assert row['pointwise_oracle_gamma_value_error_per_coordinate'] <= row['global_oracle_gamma_value_error_per_coordinate']
    for gamma in [-2., 0., .5, 1., 3.]:
        candidate = np.sum((gamma*v-u)**2, axis=-1)
        assert np.all(candidate+1e-13 >= saved['pointwise_oracle_value_error'])
