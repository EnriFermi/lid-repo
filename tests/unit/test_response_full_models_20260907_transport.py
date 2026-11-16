import math
import numpy as np
import torch
from experiments.response_full_models_20260907_transport import integrate,interpolate_gamma,calibrated_projected,prepare_fixture


def test_flat_posterior_transport_contracts_only_normal_by_noise_ratio():
    x=torch.tensor([[2.,3.,-4.],[-1.,6.,8.]],dtype=torch.float64)
    def predict(q,s):
        result=q.clone();result[:,1:]=0
        return result
    endpoint,trajectory,scales=integrate(predict,x)
    expected=x.clone();expected[:,1:]*=.01
    torch.testing.assert_close(endpoint,expected,rtol=1e-7,atol=1e-9)
    assert trajectory.shape==(5,2,3)
    np.testing.assert_allclose(scales,np.logspace(0,-2,5),rtol=1e-14)


def test_gaussian_probability_flow_has_exact_covariance_rescaling():
    x=torch.tensor([[1.,2.,3.]],dtype=torch.float64);variance=torch.tensor([.01,1.,4.],dtype=torch.float64)
    endpoint,_,_=integrate(lambda q,s:q*variance/(variance+s*s),x)
    expected=x*torch.sqrt((variance+.01**2)/(variance+1))
    torch.testing.assert_close(endpoint,expected,rtol=1e-7,atol=1e-9)


def test_frozen_logscale_interpolation_and_projected_adapter():
    grid=np.array([.01,.1,1.]);gamma=np.array([2.,1.,.5])
    assert interpolate_gamma(math.sqrt(.001),grid,gamma)==1.5
    x=torch.tensor([[1.,2.,3.]],dtype=torch.float64);b=x*.5
    center=torch.zeros(3,dtype=torch.float64);basis=torch.eye(3,dtype=torch.float64)[:,:2]
    got=calibrated_projected(b,x,center,basis,2.)
    torch.testing.assert_close(got,torch.zeros_like(x))


def test_initial_and_reference_are_canonical_noisy_population_not_raw_unit_noise():
    source={"mean":torch.zeros(30),"rms":.5}
    a=prepare_fixture("e8_spaghetti_pca",source,count=8)
    np.testing.assert_allclose(a["initial"],a["clean_raw"]/.5+a["noise"])
    np.testing.assert_allclose(a["reference_raw"],a["clean_raw"]+.005*a["noise"],atol=1e-15)
    assert np.any(a["reference_raw"][:,20:]!=0)
