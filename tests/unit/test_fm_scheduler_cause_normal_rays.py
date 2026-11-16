import pytest
import torch

from experiments.fm_mechanism_oracles import make_affine_fixture
from experiments.fm_scheduler_cause_normal_rays import normal_ray_inputs,error_components,RHOS


@pytest.mark.parametrize("law",("gaussian","uniform"))
def test_exact_oracle_invariant_along_finite_normal_rays(law):
    fixture=make_affine_fixture(kind=law,ambient_dim=8,intrinsic_dim=2,seed=43,
        center=torch.arange(8,dtype=torch.float64)/10, population_rms=None)
    generator=torch.Generator().manual_seed(17)
    observed=torch.randn(64,8,generator=generator,dtype=torch.float64)
    basis,tangent,normal,rays=normal_ray_inputs(observed,fixture.basis,fixture.center)
    torch.testing.assert_close(normal@basis,torch.zeros(64,2,dtype=torch.float64),rtol=0,atol=1e-14)
    torch.testing.assert_close(rays[1.],observed)
    truth=fixture.posterior(tangent,.316227766).mean
    for rho in RHOS:
        torch.testing.assert_close(fixture.posterior(rays[rho],.316227766).mean,truth,atol=2e-14,rtol=2e-14)


def test_orthogonal_error_decomposition_and_zero_baseline_delta():
    generator=torch.Generator().manual_seed(18)
    basis=torch.linalg.qr(torch.randn(8,2,generator=generator,dtype=torch.float64)).Q
    truth,prediction,query=torch.randn(3,64,8,generator=generator,dtype=torch.float64)
    values=error_components(prediction,truth,prediction,basis,query)
    torch.testing.assert_close(values["tangent_energy"]+values["normal_energy"],(prediction-truth).square().sum(-1))
    assert values["delta_tangent"].count_nonzero()==0
    assert values["delta_normal_energy"].count_nonzero()==0
