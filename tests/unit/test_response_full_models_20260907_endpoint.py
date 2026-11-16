import math
import pytest
import torch
from experiments.response_full_models_20260907_endpoint import head_value_jacobian,stable_terms
from experiments.response_full_models_20260907_spaghetti_local import active_jacobian
from experiments.response_full_training_20260907 import posterior
from models.training import diffusion_denoiser_output
from models.neural_fields import NeuralFieldConfig,ScaleConditionedNeuralField


@pytest.mark.parametrize("family",["ve_edm","vp_noise","vp_posterior"])
def test_exact_native_jacobian_and_stable_wrapper_match_generic_autograd(family):
    torch.manual_seed(12)
    field=ScaleConditionedNeuralField(NeuralFieldConfig(ambient_dim=4,hidden_dim=8,depth=2,condition_dim=8,
        fourier_features=4,condition_transform="log" if family=="ve_edm" else "linear",zero_init_output=False)).double().eval()
    q=torch.randn(3,4,dtype=torch.float64);scale=.13;s=1.7 if family=="ve_edm" else 1.
    den=scale**2+s**2;condition=scale if family=="ve_edm" else math.log(scale)
    native=q/math.sqrt(den)
    raw,j=head_value_jacobian(field,native,condition)
    value,autoj=active_jacobian(lambda x:field(x,condition),native,active=4)
    torch.testing.assert_close(raw,value);torch.testing.assert_close(j,autoj)
    terms=stable_terms(raw,j,q,scale,family,s)
    wrap=(lambda x:diffusion_denoiser_output(field,x,scale,preconditioning="edm",data_std=s)) if family=="ve_edm" else (lambda x:posterior(field,x,scale,family[3:]))
    b,bj=active_jacobian(wrap,q,active=4)
    torch.testing.assert_close(b,terms["posterior"])
    torch.testing.assert_close(bj,terms["canonical_jacobian_minus_identity"]+torch.eye(4)[None])
    torch.testing.assert_close(torch.diagonal(bj,dim1=-2,dim2=-1).sum(-1),terms["response"])
    torch.testing.assert_close(((b-q)/scale).square().sum(-1),terms["correction"])


@pytest.mark.parametrize("family",["ve_edm","vp_noise"])
def test_tiny_scale_stable_c_does_not_vanish_with_subtraction_cancellation(family):
    q=torch.ones(2,4,dtype=torch.float64);raw=torch.ones_like(q)*.25;j=torch.zeros(2,4,4,dtype=q.dtype)
    terms=stable_terms(raw,j,q,1e-20,family)
    torch.testing.assert_close(terms["posterior"],q,atol=0,rtol=0)
    torch.testing.assert_close(terms["correction"],torch.full((2,),.25,dtype=q.dtype))
    assert torch.all(terms["full"]>4)


def test_direct_posterior_has_no_structurally_forced_ambient_response():
    q=torch.ones(2,4,dtype=torch.float64);j=torch.eye(4,dtype=q.dtype)[None].expand(2,-1,-1)*.5
    terms=stable_terms(q*.5,j,q,1e-8,"vp_posterior")
    torch.testing.assert_close(terms["response"],torch.full((2,),2.,dtype=q.dtype))
    assert torch.all(terms["correction"]>1e15)


def test_exact_fixture_requires_canonical_coordinates_and_fixed_scale_grid():
    import numpy as np
    from experiments.response_full_models_20260907_endpoint_exact import validate_fixture
    rng=np.random.default_rng(2);raw=rng.normal(size=(64,30));mean=rng.normal(size=30);rms=.7
    fixture={"queries":(raw-mean)/rms,"query_raw":raw,"normalization_mean":mean,
        "normalization_rms":rms,"scales":np.logspace(-5,-2,13),"query_ids":np.arange(64)}
    validate_fixture(fixture,mean,rms)
    with pytest.raises(AssertionError):validate_fixture(fixture,mean,rms*2)


def test_joint_window_flags_are_pointwise_not_overlap_of_aggregate_fractions():
    import numpy as np
    from experiments.response_full_models_20260907_endpoint_comparison import accuracy_flags
    model=np.array([[1.,2.]]);oracle=np.array([[2.,1.]])
    flags=accuracy_flags(model,model,oracle,oracle)
    assert flags["model_full_good"].mean()==.5 and flags["oracle_full_good"].mean()==.5
    assert not flags["joint_full_good"].any()
