import copy
import torch
from experiments.response_full_models_20260907_rb_training import cached_edm_loss,weighted_target_loss,make_pair
from models.neural_fields import NeuralFieldConfig,ScaleConditionedNeuralField
from models.training import diffusion_ve_dsm_loss,_sample_log_uniform


def tiny_field():
    return ScaleConditionedNeuralField(NeuralFieldConfig(ambient_dim=4,hidden_dim=8,depth=1,
        condition_dim=8,fourier_features=4,condition_transform="log",zero_init_output=False)).double()


def test_cached_original_edm_loss_and_gradient_bitwise_replay_source_draws():
    torch.manual_seed(93);field=tiny_field();other=copy.deepcopy(field);x=torch.randn(7,4,dtype=torch.float64)
    one=torch.Generator().manual_seed(94);two=torch.Generator().manual_seed(94)
    original=diffusion_ve_dsm_loss(field,x,sigma_min=.01,sigma_max=16.,generator=one,loss_weighting="edm_x0",preconditioning="edm")
    sigma=_sample_log_uniform(len(x),minimum=.01,maximum=16.,data=x,generator=two)
    noise=torch.randn(x.shape,dtype=x.dtype,generator=two);observed=x+sigma[:,None]*noise
    cached=cached_edm_loss(other,observed,sigma,x)
    torch.testing.assert_close(original,cached,atol=0,rtol=0)
    original.backward();cached.backward()
    for a,b in zip(field.parameters(),other.parameters()):torch.testing.assert_close(a.grad,b.grad,atol=0,rtol=0)
    assert torch.equal(one.get_state(),two.get_state())


def test_conditional_mean_removes_bayes_constant_but_preserves_expected_raw_gradient():
    torch.manual_seed(95);prediction=torch.randn(5,4,dtype=torch.float64,requires_grad=True)
    mean=torch.randn(5,4,dtype=torch.float64);delta=torch.randn(5,4,dtype=torch.float64)
    sigma=torch.linspace(.1,1,5,dtype=torch.float64)
    noisy=.5*(weighted_target_loss(prediction,mean+delta,sigma)+weighted_target_loss(prediction,mean-delta,sigma))
    rb=weighted_target_loss(prediction,mean,sigma)
    grad_noisy=torch.autograd.grad(noisy,prediction,retain_graph=True)[0]
    grad_rb=torch.autograd.grad(rb,prediction)[0]
    torch.testing.assert_close(grad_noisy,grad_rb,rtol=1e-14,atol=1e-14)
    torch.testing.assert_close(noisy-rb,weighted_target_loss(torch.zeros_like(delta),delta,sigma))


def test_fresh_paired_optimizers_have_independent_identical_source_weights():
    field=tiny_field();fields,opts=make_pair(field,{"learning_rate":.001,"weight_decay":.0001})
    for a,b in zip(fields["sample_target"].parameters(),fields["posterior_target"].parameters()):
        assert torch.equal(a,b) and a.data_ptr()!=b.data_ptr()
    assert all(not opt.state for opt in opts.values())
