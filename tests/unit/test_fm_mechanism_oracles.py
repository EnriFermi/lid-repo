import numpy as np
import pytest
import torch

from experiments.fm_mechanism_oracles import (
    make_affine_fixture,
    uniform_posterior_moments,
)


@pytest.mark.parametrize("kind", ["gaussian", "uniform"])
def test_fixture_normalization_sampling_and_transform(kind):
    fixture = make_affine_fixture(kind, 16, 3, seed=9)
    variance_factor = 1 / 3 if kind == "uniform" else 1
    rms_squared = float(variance_factor * fixture.latent_scale.square().sum() / 16)
    assert rms_squared == pytest.approx(1)
    torch.manual_seed(19)
    expected = torch.rand(2)
    torch.manual_seed(19)
    sample = fixture.sample(4000, 13, dtype=torch.float64)
    assert torch.equal(torch.rand(2), expected)
    assert torch.equal(sample, fixture.sample(4000, 13, dtype=torch.float64))
    normal = sample - (sample @ fixture.basis) @ fixture.basis.T
    assert normal.abs().max() < 2e-14
    offset = torch.arange(16, dtype=torch.float64) / 40
    transformed = fixture.transformed(offset, 2.5)
    torch.testing.assert_close(
        transformed.sample(4000, 13, dtype=torch.float64), (sample - offset) / 2.5
    )


@pytest.mark.parametrize("kind", ["gaussian", "uniform"])
def test_posterior_jacobian_matches_autograd(kind):
    fixture = make_affine_fixture(
        kind, 8, 3,
        latent_scale=torch.tensor([1.0, 1.5, 2.0]), population_rms=None,
    )
    query = fixture.sample(3, 12, dtype=torch.float64)
    query[1] = fixture.basis[:, 0] * 7.0  # Tail of uniform support.
    sigma = 0.35
    posterior = fixture.posterior(query, sigma)
    for i in range(3):
        jacobian = torch.autograd.functional.jacobian(
            lambda q: fixture.posterior(q, sigma).mean, query[i]
        )
        torch.testing.assert_close(
            jacobian, posterior.jacobian(fixture.basis)[i], atol=2e-9, rtol=2e-8
        )
    torch.testing.assert_close(
        posterior.response,
        posterior.jacobian(fixture.basis).diagonal(dim1=-2, dim2=-1).sum(-1),
    )
    torch.testing.assert_close(posterior.full, posterior.response + posterior.correction)


def test_gaussian_closed_form_and_normal_displacement():
    fixture = make_affine_fixture(
        "gaussian", 5, 2, latent_scale=torch.tensor([2.0, 3.0]),
        population_rms=None, tangent_basis=torch.eye(5, dtype=torch.float64)[:, :2],
    )
    query = torch.tensor(
        [[1.0, 2.0, 0.0, 0.0, 0.0], [1.0, 2.0, 0.5, 0.0, 0.0]],
        dtype=torch.float64,
    )
    result = fixture.posterior(query, 0.5)
    expected_gain = torch.tensor([4 / 4.25, 9 / 9.25], dtype=torch.float64)
    torch.testing.assert_close(result.tangent_gain[0], expected_gain)
    torch.testing.assert_close(result.mean[0, :2], query[0, :2] * expected_gain)
    assert float(result.correction[1] - result.correction[0]) == pytest.approx(1.0)
    assert result.response[0] == result.response[1]


def test_uniform_center_boundary_and_extreme_tails():
    query = torch.tensor([0.0, 1.0, -1.0, 50.0, -50.0, 1e5], dtype=torch.float64)
    sigma = 0.01
    mean, variance = uniform_posterior_moments(query, sigma, -1.0, 1.0)
    assert torch.isfinite(mean).all() and torch.isfinite(variance).all()
    assert (variance > 0).all()
    assert ((mean >= -1) & (mean <= 1)).all()
    assert float(mean[0]) == pytest.approx(0, abs=1e-14)
    assert float(variance[0]) == pytest.approx(sigma**2)
    assert float(mean[1]) == pytest.approx(1 - sigma * np.sqrt(2 / np.pi), abs=1e-14)
    assert float(variance[1]) == pytest.approx(sigma**2 * (1 - 2 / np.pi), abs=1e-14)
    assert float(mean[2]) == pytest.approx(-float(mean[1]), abs=1e-14)
    assert float(variance[3]) == pytest.approx(sigma**4 / 49**2, rel=1e-6)
    assert float(variance[5]) == pytest.approx(sigma**4 / (1e5 - 1)**2, rel=1e-6)


def test_uniform_very_wide_noise_becomes_prior():
    query = torch.tensor([0.0, 0.7], dtype=torch.float64)
    mean, variance = uniform_posterior_moments(query, 1e6, -1.0, 1.0)
    torch.testing.assert_close(mean, torch.zeros_like(query), atol=3e-12, rtol=0)
    torch.testing.assert_close(variance, torch.full_like(query, 1 / 3), atol=3e-12, rtol=0)


def test_uniform_far_tail_derivative_matches_posterior_gain():
    query = torch.tensor([7.0, 50.0, 1e5], dtype=torch.float64, requires_grad=True)
    mean, variance = uniform_posterior_moments(query, 0.01, -1.0, 1.0)
    derivative = torch.autograd.grad(mean.sum(), query)[0]
    torch.testing.assert_close(derivative, variance / 0.01**2, rtol=2e-10, atol=0)


def test_uniform_face_has_exact_asymmetry_correction():
    fixture = make_affine_fixture(
        "uniform", 8, 3, population_rms=None,
        tangent_basis=torch.eye(8, dtype=torch.float64)[:, :3],
    )
    query = torch.zeros(8, dtype=torch.float64)
    query[0] = 1.0
    result = fixture.posterior(query, 0.01)
    assert float(result.response) == pytest.approx(3 - 2 / np.pi, abs=1e-12)
    assert float(result.correction) == pytest.approx(2 / np.pi, abs=1e-12)
    assert float(result.full) == pytest.approx(3.0, abs=1e-12)


def test_uniform_against_independent_dense_integration():
    grid = np.linspace(-1.0, 1.0, 400001)
    queries = [-1.2, -0.9, 0.0, 0.7, 2.0]
    sigma = 0.3
    mean, variance = uniform_posterior_moments(torch.tensor(queries), sigma, -1.0, 1.0)
    for i, query in enumerate(queries):
        mass = np.exp(-0.5 * ((grid - query) / sigma) ** 2)
        mass /= np.trapezoid(mass, grid)
        expected_mean = np.trapezoid(grid * mass, grid)
        expected_variance = np.trapezoid((grid - expected_mean) ** 2 * mass, grid)
        assert float(mean[i]) == pytest.approx(expected_mean, abs=5e-8)
        assert float(variance[i]) == pytest.approx(expected_variance, abs=5e-9)


@pytest.mark.parametrize("kind", ["gaussian", "uniform"])
def test_scale_broadcast_and_supplied_basis(kind):
    basis = torch.eye(4, dtype=torch.float64)[:, :2]
    fixture = make_affine_fixture(kind, 4, 2, tangent_basis=basis)
    query = fixture.sample(3, 1)
    sigma = torch.tensor([0.1, 0.2, 0.3])
    result = fixture.posterior(query, sigma)
    assert result.mean.shape == (3, 4)
    assert result.tangent_gain.shape == (3, 2)
    for i in range(3):
        torch.testing.assert_close(result.mean[i], fixture.posterior(query[i], sigma[i]).mean)


def test_invalid_scales_and_basis_are_rejected():
    with pytest.raises(ValueError):
        uniform_posterior_moments(torch.zeros(1), 0.0, -1, 1)
    with pytest.raises(ValueError):
        make_affine_fixture(tangent_basis=torch.ones(128, 3))
