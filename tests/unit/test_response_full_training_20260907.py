import copy
import pytest
import torch
from experiments.response_full_training_20260907 import posterior, objective, readout
from models.neural_fields import NeuralFieldConfig, ScaleConditionedNeuralField


@pytest.mark.parametrize("kind", ["posterior", "noise"])
def test_trace_and_values_match_autograd(kind):
    torch.manual_seed(71)
    model=ScaleConditionedNeuralField(NeuralFieldConfig(ambient_dim=3,hidden_dim=8,depth=2,
        condition_dim=8,fourier_features=4,zero_init_output=False)).double().eval()
    points=torch.randn(4,3,dtype=torch.float64,requires_grad=True)
    b,r,c=readout(model,points,.23,kind)
    expected=posterior(model,points,.23,kind)
    trace=sum(torch.autograd.grad(expected[:,i].sum(),points,retain_graph=True)[0][:,i] for i in range(3))
    torch.testing.assert_close(b,expected,rtol=1e-11,atol=1e-11)
    torch.testing.assert_close(r,trace,rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(c,(expected-points).square().sum(-1)/.23**2)


def test_noise_objective_is_original_posterior_objective():
    torch.manual_seed(9)
    model=ScaleConditionedNeuralField(NeuralFieldConfig(ambient_dim=3,hidden_dim=8,depth=1,
        condition_dim=8,fourier_features=4,zero_init_output=False)).double()
    x=torch.randn(7,3,dtype=torch.float64);z=torch.randn_like(x)
    lam=torch.linspace(.03,.9,7,dtype=torch.float64)
    loss=objective(model,x,z,lam,"noise")
    raw=model((x+lam[:,None]*z)/torch.sqrt(1+lam[:,None]**2),lam.log())
    expected=(raw-z).square().mean()
    torch.testing.assert_close(loss,expected)
    a=torch.autograd.grad(loss,tuple(model.parameters()),retain_graph=True)
    b=torch.autograd.grad(expected,tuple(model.parameters()))
    for first,second in zip(a,b):
        torch.testing.assert_close(first,second,rtol=1e-10,atol=1e-10)


def test_parameter_copies_and_adam_states_are_independent():
    model=ScaleConditionedNeuralField(NeuralFieldConfig(ambient_dim=3,hidden_dim=8,depth=1,
        condition_dim=8,fourier_features=4))
    other=copy.deepcopy(model)
    opts=[torch.optim.AdamW(m.parameters()) for m in [model,other]]
    for a,b in zip(model.parameters(),other.parameters()):
        assert a.data_ptr()!=b.data_ptr()
    x=torch.randn(4,3);lam=torch.ones(4)*.1;z=torch.randn_like(x)
    objective(model,x,z,lam,"posterior").backward();opts[0].step()
    assert not opts[1].state


def test_zero_noise_head_is_identity_not_zero_posterior():
    model=ScaleConditionedNeuralField(NeuralFieldConfig(ambient_dim=3,hidden_dim=8,depth=1,
        condition_dim=8,fourier_features=4)).double()
    x=torch.randn(3,3,dtype=torch.float64)
    b,r,c=readout(model,x,.17,"noise")
    torch.testing.assert_close(b,x)
    torch.testing.assert_close(r,torch.full((3,),3.,dtype=x.dtype))
    torch.testing.assert_close(c,torch.zeros(3,dtype=x.dtype))
