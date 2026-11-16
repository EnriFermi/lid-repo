import torch

from experiments.response_full_residual_calibration_20260907 import blend, fit_residual_coefficient


def test_unconstrained_fit_keeps_negative_and_large_steps():
    a=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float64)
    b=a+torch.tensor([[1.,-.2],[.4,.5]],dtype=torch.float64)
    for coefficient in [-2.,0.,.4,1.,3.]:
        target=blend(a,b,coefficient)
        eta=fit_residual_coefficient(a,b,target)
        torch.testing.assert_close(eta,torch.tensor(coefficient,dtype=a.dtype))
        torch.testing.assert_close(blend(a,b,eta),target)


def test_least_squares_stationarity_and_zero_step():
    rng=torch.Generator().manual_seed(13)
    a,b,target=[torch.randn(17,3,generator=rng,dtype=torch.float64) for _ in range(3)]
    eta=fit_residual_coefficient(a,b,target)
    assert abs(float(((b-a)*(blend(a,b,eta)-target)).sum())) < 1e-12
    loss=lambda e:float((blend(a,b,e)-target).square().mean())
    assert loss(eta)<=min(loss(0),loss(1))
    assert fit_residual_coefficient(a,a,target)==0


def test_actual_composed_field_trace_is_linear_but_correction_is_not():
    q=torch.tensor([.2,-.7],dtype=torch.float64)
    f=lambda x:torch.stack((x[0]**2+x[1],torch.sin(x[0])+x[1]**2))
    eta=.4
    first=f(q);second=f(first)
    j1=torch.func.jacrev(f)(q)
    j2=torch.func.jacrev(lambda x:f(f(x)))(q)
    jg=torch.func.jacrev(lambda x:blend(f(x),f(f(x)),eta))(q)
    torch.testing.assert_close(jg.trace(),blend(j1.trace(),j2.trace(),eta))
    direct=(blend(first,second,eta)-q).square().sum()
    wrong=blend((first-q).square().sum(),(second-q).square().sum(),eta)
    assert not torch.isclose(direct,wrong)
