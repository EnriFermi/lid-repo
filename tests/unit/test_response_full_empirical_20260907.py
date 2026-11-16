import torch

from experiments.response_full_empirical_20260907 import empirical_posterior


def test_exact_two_point_symmetry():
    x=torch.tensor([[-1.,0.],[1.,0.]],dtype=torch.float64)
    q=torch.tensor([[0.,0.]],dtype=torch.float64)
    out=empirical_posterior(x,q,2.)
    torch.testing.assert_close(out['posterior'],q)
    torch.testing.assert_close(out['response'],torch.tensor([.25],dtype=torch.float64))
    torch.testing.assert_close(out['effective_points'],torch.tensor([2.],dtype=torch.float64))


def test_finite_difference_trace_and_moment_identity():
    x=torch.tensor([[-1.,.2],[1.,0.],[.3,.8]],dtype=torch.float64)
    q=torch.tensor([[.4,.1]],dtype=torch.float64)
    scale=.45
    out=empirical_posterior(x,q,scale)
    trace=0.
    for j in range(2):
        eps=torch.zeros_like(q);eps[0,j]=1e-5
        plus=empirical_posterior(x,q+eps,scale)['posterior']
        minus=empirical_posterior(x,q-eps,scale)['posterior']
        trace+=(plus[0,j]-minus[0,j])/2e-5
    torch.testing.assert_close(out['response'][0],trace,atol=1e-8,rtol=1e-8)
    torch.testing.assert_close(out['response']+out['correction'],out['full'])


def test_single_training_atom_has_zero_response():
    train=torch.tensor([[4.,3.]],dtype=torch.float64)
    query=torch.tensor([[1.,2.]],dtype=torch.float64)
    out=empirical_posterior(train,query,.01)
    torch.testing.assert_close(out['response'],torch.zeros(1,dtype=torch.float64))
    torch.testing.assert_close(out['posterior'],train)
