import numpy as np
import torch

from experiments.response_full_rb_cache_20260907 import grid_posterior,phase_grid,required_nodes
from experiments.response_full_oracle_20260907_spaghetti import spaghetti_posterior


def test_phase_grid_normalization_and_exact_origin_symmetry():
    torch.set_num_threads(2)
    b=grid_posterior(torch.zeros((1,20),dtype=torch.float64),torch.tensor([.6]),phase_grid(4096))
    torch.testing.assert_close(b,torch.zeros_like(b),atol=1e-13,rtol=0)


def test_noisy_phase_mean_matches_independent_adaptive_integral():
    torch.set_num_threads(2)
    k=np.arange(2,22); q=np.sin(.718*k)+np.random.default_rng(37).normal(size=20)*.08
    sigma=.08; n=int(required_nodes(sigma))
    actual=grid_posterior(torch.tensor(q)[None,:],torch.tensor([sigma],dtype=torch.float64),phase_grid(n))[0]
    expected=spaghetti_posterior(q,sigma,order=16,cutoff=14.).mean
    np.testing.assert_allclose(actual.numpy(),expected,atol=2e-12)


def test_resolution_monotone_power_of_two_and_lower_bound():
    scales=np.array([.0057,.02,.1,1.,10.])
    n=required_nodes(scales)
    assert np.all(np.diff(n)<=0) and np.all(n>=512)
    assert np.all(np.log2(n)==np.floor(np.log2(n)))
