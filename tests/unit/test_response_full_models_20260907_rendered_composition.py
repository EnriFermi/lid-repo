import numpy as np
import torch
from experiments.response_full_models_20260907_rendered_composition import exact_two_pass,estimated_seconds
from experiments.spiral_spaghetti_convergence import posterior,exact_trace


class Field(torch.nn.Module):
    def __init__(self,nonlinear=False):
        super().__init__();self.linear=torch.nn.Linear(4,4,dtype=torch.float64);self.nonlinear=nonlinear
    def forward(self,x,condition):
        y=self.linear(x)+.07*condition[:,None]
        return torch.sin(y) if self.nonlinear else y


def test_native_rectified_affine_twopass_has_total_alpha_squared_trace():
    torch.manual_seed(916);field=Field();q=torch.randn(5,4,dtype=torch.float64);scale=.3162;alpha=1/(1+scale)
    result=exact_two_pass(field,q,scale,2)
    expected_r1=alpha*torch.trace(field.linear.weight);expected_r2=alpha**2*torch.trace(field.linear.weight@field.linear.weight)
    torch.testing.assert_close(result['base_response'],expected_r1.expand(5));torch.testing.assert_close(result['response'],expected_r2.expand(5))
    torch.testing.assert_close(result['correction'],((result['posterior']-q)/scale).square().sum(-1))


def test_nonlinear_actual_composed_trace_matches_coordinate_autograd_and_old_backend():
    torch.manual_seed(552);field=Field(True);q=torch.randn(5,4,dtype=torch.float64);scale=.01
    result=exact_two_pass(field,q,scale,2);b,r=exact_trace(field,q,scale)
    torch.testing.assert_close(result['base_posterior'],b);torch.testing.assert_close(result['base_response'],r)
    for i in range(5):
        x=q[i].requires_grad_(True)
        jac=torch.autograd.functional.jacobian(lambda a:posterior(field,posterior(field,a[None],scale),scale)[0],x)
        torch.testing.assert_close(torch.trace(jac),result['response'][i])
    other=exact_two_pass(field,q,scale,4)
    for key in result:torch.testing.assert_close(result[key],other[key],atol=1e-12,rtol=1e-12)


def test_original_float32_query_normalization_is_preserved_before_promotion():
    raw=np.array([[.123456789,1.234567891]],dtype=np.float64)
    mean=torch.tensor([.0112233,.5544332],dtype=torch.float32);rms=torch.tensor(.3812771,dtype=torch.float32)
    actual=((torch.from_numpy(np.array(raw,dtype=np.float32))-mean)/rms).double()
    original=(torch.from_numpy(raw.astype(np.float32))-mean)/rms
    assert torch.equal(actual.float(),original)


def test_declared_cost_estimate_expands_two_query_pilot_without_trace_approximation():
    assert estimated_seconds(.5)==288
    assert estimated_seconds(2.)>600


def test_version2_preserves_paired_float64_operator_bitwise():
    from experiments.response_full_models_20260907_rendered_composition_v2 import exact_two_pass as version2
    torch.manual_seed(552)
    field=Field(True);q=torch.randn(5,4,dtype=torch.float64)
    for scale in (.01,.3162,1.):
        original=exact_two_pass(field,q,scale,2)
        revised=version2(field,q,scale,2)
        for key in original:
            assert torch.equal(original[key],revised[key])
