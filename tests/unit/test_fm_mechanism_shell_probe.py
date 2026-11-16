import math

import pytest

torch = pytest.importorskip("torch")

from experiments.fm_mechanism_shell_probe import make_population, oracle_at


@pytest.mark.parametrize("law", ["gaussian", "uniform"])
def test_affine_oracle_ignores_normal_sweep(law):
    generator = torch.Generator().manual_seed(42)
    basis = torch.linalg.qr(torch.randn(9, 3, generator=generator, dtype=torch.float64)).Q
    fixture = {"data_law": law, "tangent_basis": basis, "latent_std": 2., "latent_half_width": 3.}
    population, clean, frame, zt, zn, direction = make_population(fixture, 11, 932)
    torch.testing.assert_close(torch.einsum("ndi,nd->ni", frame, zn), torch.zeros(11, 3, dtype=torch.float64), atol=1e-14, rtol=0)
    for tangent_noise in (0, 1):
        reference, _, _ = oracle_at(clean+.316*zt*tangent_noise, .316, fixture, population, frame)
        for rho in (.25, .5, 1.):
            actual, _, _ = oracle_at(clean+.316*(zt*tangent_noise+rho*zn), .316, fixture, population, frame)
            torch.testing.assert_close(actual, reference, atol=1e-13, rtol=1e-13)
    expected_noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(932+1701), dtype=torch.float64)
    torch.testing.assert_close(zt+zn, expected_noise, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(direction.norm(dim=-1), torch.ones(11, dtype=torch.float64))


def test_circle_frame_and_oracle_jacobian_projection():
    fixture = {"data_law": "circle", "circle_radius": math.sqrt(2)}
    population, clean, frame, zt, zn, direction = make_population(fixture, 11, 93)
    torch.testing.assert_close(clean.norm(dim=-1), torch.full((11,), math.sqrt(2), dtype=torch.float64))
    torch.testing.assert_close((frame[..., 0]*clean).sum(-1), torch.zeros(11, dtype=torch.float64), atol=1e-14, rtol=0)
    values, jtt, jac = oracle_at(clean+.316*(zt+zn), .316, fixture, population, frame)
    assert values.shape == clean.shape
    assert jtt.shape == (11, 1, 1)
    torch.testing.assert_close(jtt[:, 0, 0], torch.einsum("ni,nij,nj->n", frame[..., 0], jac, frame[..., 0]))


def test_original_gaussian_fixture_without_data_law_is_supported():
    fixture = {"tangent_basis": torch.eye(4, dtype=torch.float64)[:, :2], "latent_std": 1.}
    population, clean, *_ = make_population(fixture, 5, 9)
    assert population.kind == "gaussian"
    assert clean.shape == (5, 4)
