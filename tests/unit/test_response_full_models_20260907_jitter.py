import torch

from experiments.response_full_models_20260907_jitter import antithetic_readout


def test_affine_posterior_and_full_are_exactly_preserved_by_any_antithetic_tape():
    torch.manual_seed(78)
    matrix=torch.randn(5,5,dtype=torch.float64);offset=torch.randn(5,dtype=torch.float64)
    q=torch.randn(4,5,dtype=torch.float64);z=torch.randn(4,8,5,dtype=torch.float64)
    result=antithetic_readout(lambda x:x@matrix.T+offset,q,.2,z,8)
    target=q@matrix.T+offset
    torch.testing.assert_close(result['posterior'],target)
    torch.testing.assert_close(result['response'],torch.full((4,),matrix.trace(),dtype=q.dtype))
    torch.testing.assert_close(result['correction_original'],(target-q).square().sum(-1)/.2**2)


def test_nonlinear_response_is_derivative_of_averaged_function_not_averaged_full():
    torch.manual_seed(79)
    q=torch.randn(3,4,dtype=torch.float64);z=torch.randn(3,8,4,dtype=torch.float64)
    result=antithetic_readout(torch.sin,q,.3,z,8)
    inputs=q.detach().requires_grad_(True)
    g=(torch.sin(inputs[:,None]+.3*z)+torch.sin(inputs[:,None]-.3*z)).mean(1)/2
    trace=sum(torch.autograd.grad(g[:,i].sum(),inputs,retain_graph=True)[0][:,i] for i in range(4))
    torch.testing.assert_close(result['response'],trace)
    torch.testing.assert_close(result['full'],trace+(g-inputs).square().sum(-1)/.3**2)
