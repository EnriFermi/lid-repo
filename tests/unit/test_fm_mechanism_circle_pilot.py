import math

import numpy as np
import pytest
import torch

from experiments.fm_mechanism_pilot import PilotConfig, make_fixture, population_posterior
from experiments.ve_torus_knee import circle_oracle_curve


def test_circle_fixture_and_posterior():
    cfg = PilotConfig(data_law="circle", ambient_dim=2, intrinsic_dim=1,
                      validation_count=64, device="cpu")
    data = make_fixture(cfg)
    clean = data["clean_queries"].double()
    assert torch.allclose(clean.norm(dim=-1), torch.full((64,), math.sqrt(2), dtype=torch.float64), atol=2e-7)
    scales = [.01, .1, .316227766, 1.0]
    oracle = circle_oracle_curve(scales, math.sqrt(2))
    for lam, expected_cosine in zip(scales, oracle["posterior_cosine"]):
        predicted = population_posterior(clean, lam, data)
        assert np.isclose(float(predicted.norm(dim=-1).mean()), math.sqrt(2)*expected_cosine, atol=2e-7)
        zero = population_posterior(torch.zeros(1, 2), lam, data)
        assert torch.equal(zero, torch.zeros_like(zero))


def test_circle_requires_two_ambient_dimensions():
    with pytest.raises(ValueError, match="D=2"):
        PilotConfig(data_law="circle", ambient_dim=3, intrinsic_dim=1)
