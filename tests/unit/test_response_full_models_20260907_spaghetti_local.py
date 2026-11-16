import numpy as np
import torch

from experiments.response_full_models_20260907_spaghetti_local import tangent_from_phase,split_jacobian_error,active_jacobian


def test_tangent_is_normalized_curve_derivative():
    phase=np.array([.1,.5,2.])
    frequencies=np.arange(2,22)
    step=1e-6
    derivative=(np.sin((phase[:,None]+step)*frequencies)-np.sin((phase[:,None]-step)*frequencies))/(2*step)
    derivative/=np.linalg.norm(derivative,axis=-1,keepdims=True)
    np.testing.assert_allclose(tangent_from_phase(phase),derivative,atol=1e-10,rtol=1e-10)


def test_nonsymmetric_jacobian_error_splits_orthogonally():
    rng=np.random.default_rng(76)
    error=rng.normal(size=(4,7,7));v=rng.normal(size=(4,7));v/=np.linalg.norm(v,axis=-1,keepdims=True)
    terms=split_jacobian_error(error,v)
    np.testing.assert_allclose(sum(terms.values()),(error**2).sum((-1,-2)),atol=1e-12)


def test_active_output_vjps_recover_nonlinear_input_columns():
    torch.manual_seed(77)
    matrix=torch.randn(7,7,dtype=torch.float64);x=torch.randn(4,7,dtype=torch.float64)
    values,jac=active_jacobian(lambda y:torch.sin(y@matrix.T),x,active=3,batch_size=2)
    expected=torch.cos(x@matrix.T)[:,:3,None]*matrix[None,:3]
    torch.testing.assert_close(jac,expected)
