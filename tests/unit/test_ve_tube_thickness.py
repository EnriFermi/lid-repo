from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.ve_tube_thickness import (
    centered_directional_gains,
    estimate_thickness_from_gains,
    gaussian_tube_oracle_curve,
    sample_gaussian_tube,
    strongest_downward_knee,
)


def test_gaussian_tube_samples_are_antithetically_centered() -> None:
    samples = sample_gaussian_tube(
        10_000,
        ambient_dim=8,
        tangent_dim=2,
        tangent_std=1.0,
        normal_std=0.1,
        seed=7,
    )

    assert samples.shape == (10_000, 8)
    np.testing.assert_allclose(samples.mean(axis=0), 0.0, atol=2.0e-7)
    assert np.std(samples[:, 0]) == pytest.approx(1.0, rel=0.04)
    assert np.std(samples[:, 7]) == pytest.approx(0.1, rel=0.04)


def test_gaussian_tube_normal_knee_equals_its_width() -> None:
    scales = np.geomspace(0.005, 2.0, 1001)
    oracle = gaussian_tube_oracle_curve(
        scales,
        ambient_dim=16,
        tangent_dim=2,
        tangent_std=3.0,
        normal_std=0.2,
    )

    knee = strongest_downward_knee(
        scales, oracle["normal"], lower=0.02, upper=1.0, smooth=False
    )

    assert knee["sigma"] == pytest.approx(0.2, rel=0.01)
    assert knee["boundary_hit"] is False


def test_one_batched_forward_recovers_gaussian_thickness() -> None:
    delta = 0.2

    class CountingGaussianDenoiser:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, inputs, sigma):  # type: ignore[no-untyped-def]
            self.calls += 1
            gain = delta**2 / (delta**2 + sigma.square())
            return gain[:, None] * inputs

    denoiser = CountingGaussianDenoiser()
    center = torch.zeros(4)
    directions = torch.eye(4)[:3]
    scales = np.geomspace(0.03, 0.8, 11)

    gains = centered_directional_gains(
        denoiser,
        center,
        directions,
        scales,
        amplitude_ratio=0.1,
    )
    estimate = estimate_thickness_from_gains(scales, gains)

    assert denoiser.calls == 1
    assert gains.shape == (11, 3)
    assert estimate["delta_hat"] == pytest.approx(delta, rel=2.0e-6)
    assert estimate["gain_fit_rmse"] == pytest.approx(0.0, abs=2.0e-7)
    assert estimate["inverse_curvature_slope"] == pytest.approx(1.0, rel=2.0e-5)
    assert estimate["inverse_curvature_intercept"] == pytest.approx(
        delta**2, rel=2.0e-5
    )


def test_thickness_estimator_abstains_without_transition_gains() -> None:
    scales = np.geomspace(0.03, 0.8, 11)
    gains = np.full((11, 3), 0.999)

    estimate = estimate_thickness_from_gains(scales, gains)

    assert estimate["delta_hat"] is None
    assert estimate["valid_count"] == 0
    assert estimate["valid_fraction"] == 0.0
