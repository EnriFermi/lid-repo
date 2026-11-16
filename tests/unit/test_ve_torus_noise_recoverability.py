from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.ve_torus_noise_recoverability import (
    antithetic_noise_recoverability,
    exploratory_sparse_selector,
)


def test_antithetic_recoverability_matches_gaussian_tube() -> None:
    delta = 0.2
    sigma = 0.3
    queries = torch.zeros((1, 4), dtype=torch.float32)
    noise = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4)

    def denoiser(inputs, conditions):  # type: ignore[no-untyped-def]
        gain = delta**2 / (delta**2 + conditions.square())
        output = inputs.clone()
        output[:, 2:] = gain[:, None] * inputs[:, 2:]
        return output

    values, calls = antithetic_noise_recoverability(
        denoiser,
        queries,
        noise,
        sigma,
        batch_size=32,
    )
    removed_normal_fraction = sigma**2 / (delta**2 + sigma**2)

    assert calls == 1
    np.testing.assert_allclose(values["factor_0"], 0.0, atol=1.0e-7)
    np.testing.assert_allclose(
        values["factor_1"], removed_normal_fraction, atol=1.0e-7
    )
    np.testing.assert_allclose(
        values["total"], 0.5 * removed_normal_fraction, atol=1.0e-7
    )


def test_sparse_selector_finds_first_saturation_before_next_transition() -> None:
    amplitude = 0.5
    delta = 0.005
    scales = [0.01, 0.02, 0.04, 0.08, 0.16]
    values = [
        amplitude * sigma**2 / (delta**2 + sigma**2) for sigma in scales[:-1]
    ] + [0.65]
    rows = [
        {
            "sigma": sigma,
            "sparse_anchor": True,
            "learned_total_mean": value,
        }
        for sigma, value in zip(scales, values)
    ]

    result = exploratory_sparse_selector(rows)

    assert result["status"] == "selected"
    assert result["scale_evaluations"] == 5
    assert result["selected_sigma"] == pytest.approx(0.04)
    assert result["fit"]["delta"] == pytest.approx(delta)
