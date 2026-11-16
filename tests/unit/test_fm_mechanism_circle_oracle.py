import math

import numpy as np
import pytest
import torch

from experiments.fm_mechanism_circle_oracle import (
    bessel_ratio_terms,
    circle_posterior,
    circle_value_and_jacobian,
)
from experiments.ve_torus_knee import circle_oracle_curve


def test_clean_circle_matches_existing_periodic_quadrature():
    radius = math.sqrt(2)
    scales = np.array([.01, .05, .1, .316227766, 1., 3.])
    previous = circle_oracle_curve(scales, radius)
    points = torch.tensor([[radius, 0.]] * len(scales), dtype=torch.float64)
    actual = circle_posterior(points, torch.from_numpy(scales), radius=radius)
    np.testing.assert_allclose(actual.response.numpy(), previous["response"], atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(actual.correction.numpy(), previous["bias"], atol=1e-12, rtol=1e-8)
    np.testing.assert_allclose(actual.full.numpy(), previous["full"], atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("point", [[0., 0.], [.2, -.4], [2., 1.], [math.sqrt(2), 0.]])
def test_arbitrary_query_jacobian_matches_autograd_and_finite_difference(point):
    query = torch.tensor(point, dtype=torch.float64)
    sigma = .35
    result = circle_posterior(query, sigma)
    jacobian = torch.autograd.functional.jacobian(lambda x: circle_posterior(x, sigma).mean, query)
    torch.testing.assert_close(jacobian, result.jacobian, rtol=2e-10, atol=3e-10)
    step = 1e-5
    eye = torch.eye(2, dtype=query.dtype)
    finite_difference = torch.stack([
        (circle_posterior(query + step*direction, sigma).mean
         - circle_posterior(query - step*direction, sigma).mean)/(2*step)
        for direction in eye
    ], dim=1)
    torch.testing.assert_close(finite_difference, result.jacobian, rtol=2e-6, atol=1e-8)
    torch.testing.assert_close(result.response, result.jacobian.trace())
    torch.testing.assert_close(result.full, result.response + result.correction)


def test_origin_is_isotropic_and_can_have_gain_above_one():
    origin = torch.zeros(2, dtype=torch.float64)
    result = circle_posterior(origin, .1, radius=2.)
    torch.testing.assert_close(result.mean, origin)
    torch.testing.assert_close(result.jacobian, torch.eye(2, dtype=torch.float64)*200)
    assert float(result.full) == pytest.approx(400)
    assert float(result.radial_gain) == pytest.approx(200)
    assert float(result.tangent_gain) == pytest.approx(200)


def test_tiny_scale_retains_curvature_normal_gain_without_cancellation():
    radius = math.sqrt(2)
    sigmas = torch.tensor([1e-2, 1e-4, 1e-6], dtype=torch.float64)
    query = torch.tensor([[radius, 0.]]*3, dtype=torch.float64)
    result = circle_posterior(query, sigmas)
    assert torch.isfinite(result.jacobian).all()
    assert (result.radial_gain > 0).all()
    torch.testing.assert_close(result.radial_gain, sigmas.square()/(2*radius**2), rtol=1e-4, atol=0)
    torch.testing.assert_close(result.tangent_gain, 1-sigmas.square()/(2*radius**2), rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(result.response, torch.ones(3, dtype=torch.float64), rtol=1e-8, atol=1e-8)
    # Curvature induces an inward displacement even for an exact, thin circle.
    assert float(result.mean[0, 0]) < radius
    assert float(result.correction[0]) > 0
    torch.testing.assert_close(result.jacobian[:, 0, 0], result.radial_gain, rtol=1e-14, atol=0)


def test_arbitrary_noisy_queries_match_direct_angular_covariance():
    radius = math.sqrt(2)
    theta = torch.arange(32768, dtype=torch.float64) * (2 * math.pi / 32768)
    support = radius * torch.stack([theta.cos(), theta.sin()], dim=1)
    for query, sigma in [(torch.tensor([.15, -.4], dtype=torch.float64), .7),
                         (torch.tensor([2.4, 1.1], dtype=torch.float64), .2)]:
        weights = torch.softmax(-((support-query).square().sum(1))/(2*sigma**2), dim=0)
        mean = (weights[:, None]*support).sum(0)
        centered = support-mean
        covariance = torch.einsum("n,ni,nj->ij", weights, centered, centered)
        actual = circle_posterior(query, sigma, radius=radius)
        torch.testing.assert_close(actual.mean, mean, rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(actual.jacobian, covariance/sigma**2, rtol=1e-9, atol=1e-11)


def test_translation_rotation_and_per_query_broadcast():
    center = torch.tensor([.3, -.4], dtype=torch.float64)
    queries = torch.tensor([[.2, .8], [-1., .6]], dtype=torch.float32)
    scales = torch.tensor([[.5], [1.]], dtype=torch.float64)
    reference = circle_posterior(queries, scales)
    shifted = circle_posterior(queries.double()+center, scales, center=center)
    torch.testing.assert_close(shifted.mean, reference.mean+center)
    torch.testing.assert_close(shifted.jacobian, reference.jacobian)
    rotation = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64)
    rotated = circle_posterior(queries.double() @ rotation.T, scales)
    torch.testing.assert_close(rotated.mean, reference.mean @ rotation.T)
    torch.testing.assert_close(rotated.jacobian, rotation @ reference.jacobian @ rotation.T)
    assert shifted.mean.dtype == torch.float64
    value, jac = circle_value_and_jacobian(queries, scales)
    torch.testing.assert_close(value, reference.mean)
    torch.testing.assert_close(jac, reference.jacobian)


def test_ratio_small_and_large_switches_are_continuous_and_consistent():
    kappa = torch.tensor([0., 1e-8, .00099999, .00100001, 49.99999, 50.00001, 1e6, 1e12], dtype=torch.float64)
    ratio, derivative, divided = bessel_ratio_terms(kappa)
    assert torch.isfinite(ratio).all() and (derivative > 0).all()
    assert float(divided[0]) == pytest.approx(.5)
    assert float(derivative[0]) == pytest.approx(.5)
    torch.testing.assert_close(ratio, kappa*divided, rtol=1e-14, atol=1e-15)
    moderate = kappa[2:6]
    direct = torch.special.i1e(moderate)/torch.special.i0e(moderate)
    torch.testing.assert_close(ratio[2:6], direct, rtol=1e-13, atol=1e-14)
    torch.testing.assert_close(derivative[-2:], .5/kappa[-2:].square(), rtol=2e-6, atol=0)


@pytest.mark.parametrize("sigma", [0., -1., float("nan"), float("inf")])
def test_invalid_noise_is_rejected(sigma):
    with pytest.raises(ValueError):
        circle_posterior(torch.zeros(2), sigma)
