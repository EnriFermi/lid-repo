from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from experiments.ve_torus_knee import (
    circle_oracle_curve,
    half_retention_crossing,
    log_slope_knee,
    sample_product_torus,
)


def test_product_torus_samples_are_centered_and_have_fixed_norm() -> None:
    samples = sample_product_torus(2_000, [1.0, 16.0], seed=7)

    assert samples.shape == (2_000, 4)
    np.testing.assert_allclose(samples.mean(axis=0), 0.0, atol=2.0e-6)
    np.testing.assert_allclose(
        np.sum(np.square(samples), axis=1), 1.0 + 16.0**2, rtol=1.0e-6
    )


def test_circle_oracle_is_scale_equivariant() -> None:
    relative_scales = np.geomspace(0.03, 20.0, 41)
    unit = circle_oracle_curve(relative_scales, 1.0, quadrature_points=16_384)
    rescaled = circle_oracle_curve(
        7.0 * relative_scales, 7.0, quadrature_points=16_384
    )

    for name in ("posterior_cosine", "response", "bias", "full"):
        np.testing.assert_allclose(unit[name], rescaled[name], rtol=1.0e-11, atol=1.0e-11)
    np.testing.assert_allclose(unit["full"], unit["response"] + unit["bias"])


def test_two_scale_oracle_has_two_geometric_knees() -> None:
    scales = np.geomspace(0.01, 40.0, 301)
    radii = (0.125, 2.0)
    factors = [
        circle_oracle_curve(scales, radius, quadrature_points=32_768)
        for radius in radii
    ]

    knee_ratios = []
    for radius, factor in zip(radii, factors, strict=True):
        knee, _ = log_slope_knee(
            scales,
            factor["full"],
            lower=0.5 * radius,
            upper=3.0 * radius,
        )
        knee_ratios.append(knee / radius)
        crossing = half_retention_crossing(scales, factor["posterior_cosine"])
        assert crossing / radius == pytest.approx(0.936, rel=0.03)

    assert knee_ratios[0] == pytest.approx(1.307, rel=0.04)
    assert knee_ratios[1] == pytest.approx(1.307, rel=0.04)
    assert knee_ratios[0] == pytest.approx(knee_ratios[1], rel=0.04)

    total = factors[0]["full"] + factors[1]["full"]
    low = int(np.argmin(np.abs(scales - 0.02)))
    middle = int(np.argmin(np.abs(scales - 0.7)))
    high = int(np.argmin(np.abs(scales - 16.0)))
    assert total[low] == pytest.approx(2.0, abs=0.02)
    assert total[middle] == pytest.approx(1.0, abs=0.12)
    assert total[high] < 0.04


def test_invalid_odd_sample_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="even"):
        sample_product_torus(5, [1.0], seed=0)
