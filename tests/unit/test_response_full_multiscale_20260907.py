import torch

from experiments.response_full_multiscale_20260907 import first_order_correction, project


def test_linear_quadratic_displacement_separation():
    x = torch.tensor([[2., 3.]], dtype=torch.float64)
    a, b, scale = torch.tensor([[.4, -.3]]), torch.tensor([[2., 1.]]), .125
    at_scale = x + a*scale + b*scale**2
    at_half = x + a*(scale/2) + b*(scale/2)**2
    torch.testing.assert_close(first_order_correction(at_scale, at_half, x, scale), a.double().square().sum(-1))


def test_constant_error_amplified_not_removed():
    x = torch.zeros(2,3, dtype=torch.float64)
    error = torch.ones_like(x)*.05
    torch.testing.assert_close(first_order_correction(x+error,x+error,x,.2), 9*error.square().sum(-1)/.2**2)


def test_project_canonical_affine_posterior():
    center = torch.tensor([1.,2.,3.], dtype=torch.float64)
    basis = torch.tensor([[1.], [0.], [0.]], dtype=torch.float64)
    value = torch.tensor([[4.,5.,6.]], dtype=torch.float64)
    torch.testing.assert_close(project(value,center,basis), torch.tensor([[4.,2.,3.]], dtype=torch.float64))
