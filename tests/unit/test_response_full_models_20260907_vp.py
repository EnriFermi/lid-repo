import torch

from experiments.response_full_models_20260907_span import projected_trace
from experiments.response_full_training_20260907 import posterior


def test_noise_canonical_projection_retains_analytic_identity_in_spatial_derivative():
    class Linear(torch.nn.Module):
        def forward(self, y, log_scale):
            return 2*y+1
    x = torch.randn(5, 6, dtype=torch.float64)
    lam = .23
    basis = torch.eye(6, dtype=torch.float64)[:, :2]
    fn = lambda q: posterior(Linear(), q, lam, "noise")
    values, trace = projected_trace(fn, x, basis, 2)
    gain = 1-2*lam/(1+lam**2)**.5
    torch.testing.assert_close(values, gain*x-lam)
    torch.testing.assert_close(trace, torch.full((5,), 2*gain, dtype=x.dtype))


def test_zero_noise_predictor_is_identity_not_zero_posterior():
    class Zero(torch.nn.Module):
        def forward(self, y, log_scale):
            return y*0
    x = torch.randn(3, 7, dtype=torch.float64)
    value, trace = projected_trace(lambda q: posterior(Zero(), q, .1, "noise"),
        x, torch.eye(7, dtype=x.dtype), 2)
    torch.testing.assert_close(value, x)
    torch.testing.assert_close(trace, torch.full((3,), 7., dtype=x.dtype))
