import numpy as np

from experiments.ve_known_lid_recoverability_campaign import (
    DATASET_SPECS,
    SELECTOR_ANCHORS,
    frozen_sparse_selector,
    sample_benchmark_coefficients,
)


def test_benchmark_coefficient_samplers_are_deterministic_and_structural() -> None:
    for name, spec in DATASET_SPECS.items():
        first = sample_benchmark_coefficients(name, 256, seed=17)
        second = sample_benchmark_coefficients(name, 256, seed=17)
        assert first.shape == (256, 30)
        assert first.dtype == np.float32
        assert np.isfinite(first).all()
        assert np.array_equal(first, second)
        assert np.all(first[:, spec.active_dim :] == 0.0)
        assert np.any(first[:, : spec.active_dim] != 0.0)


def test_frozen_selector_is_ambient_invariant_in_dimension_units() -> None:
    scales = np.asarray(SELECTOR_ANCHORS)
    delta = 0.005
    first = 2.0 * scales**2 / (delta**2 + scales**2)
    transition = np.where(scales >= 0.16, 0.8, 0.0)
    dimension_values = first + transition

    selected_four = frozen_sparse_selector(scales, dimension_values / 4.0, ambient_dim=4)
    selected_thirty = frozen_sparse_selector(
        scales, dimension_values / 30.0, ambient_dim=30
    )
    assert selected_four["status"] == "selected"
    assert selected_thirty["status"] == "selected"
    assert selected_four["selected_sigma"] == selected_thirty["selected_sigma"]


def test_frozen_selector_abstains_without_a_second_transition() -> None:
    scales = np.asarray(SELECTOR_ANCHORS)
    values = 25.0 * scales**2 / (0.006**2 + scales**2)
    result = frozen_sparse_selector(scales, values / 30.0, ambient_dim=30)
    assert result["status"] == "abstained"
