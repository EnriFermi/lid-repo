import copy
import torch
from experiments.response_full_models_20260907_clean_anchor import clean_anchor_loss,resumed_pair
from experiments.response_full_models_20260907_rb_training import cached_edm_loss
from experiments.response_full_models_20260907_fixed_objective_lr import resumed_pair as previous_pair
from models.training import diffusion_denoiser_output
from tests.unit.test_response_full_models_20260907_rb_training import tiny_field


def test_anchor_scalar_and_gradient_equal_masked_allrow_loss_not_selected_mean():
    torch.manual_seed(431);field=tiny_field().double()
    clean=torch.randn(7,4,dtype=torch.float64)
    sigma=torch.tensor([.01,.02,.03,.031,.1,1.,16.],dtype=torch.float64)
    actual=clean_anchor_loss(field,clean,sigma)
    b=diffusion_denoiser_output(field,clean,sigma,preconditioning="edm",data_std=1.)
    per=(b-clean).square().mean(1)*(1+sigma.square())/sigma.square()
    expected=(per*(sigma<=.03)).mean()
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    ga=torch.autograd.grad(actual,tuple(field.parameters()),retain_graph=True)
    ge=torch.autograd.grad(expected,tuple(field.parameters()))
    for a,e in zip(ga,ge):torch.testing.assert_close(a,e,rtol=1e-11,atol=1e-11)
    selected=cached_edm_loss(field,clean[:3],sigma[:3],clean[:3])
    torch.testing.assert_close(actual,selected*3/7)


def test_empty_anchor_is_zero_with_zero_parameter_gradients():
    field=tiny_field().double();clean=torch.randn(4,4,dtype=torch.float64);sigma=torch.ones(4,dtype=torch.float64)
    loss=clean_anchor_loss(field,clean,sigma);assert float(loss)==0
    for g in torch.autograd.grad(loss,tuple(field.parameters())):assert torch.count_nonzero(g)==0


def setup():
    torch.manual_seed(83);field=tiny_field().float()
    y=torch.randn(7,4);target=torch.randn_like(y);sigma=torch.linspace(.01,1,7)
    opt=torch.optim.AdamW(field.parameters(),lr=2e-5,weight_decay=.0001)
    cached_edm_loss(field,y,sigma,target).backward();opt.step()
    return field,opt,(y,sigma,target)


def test_resumed_pair_preserves_all_weights_moments_and_lr_without_aliases():
    field,opt,_=setup();fields,opts=resumed_pair(field,opt.state_dict())
    for arm in fields:
        assert opts[arm].param_groups[0]["lr"]==2e-5
        for old,new in zip(field.parameters(),fields[arm].parameters()):
            assert torch.equal(old,new) and old.data_ptr()!=new.data_ptr()
        for old,new in zip(opt.state.values(),opts[arm].state.values()):
            for k in ("step","exp_avg","exp_avg_sq"):
                assert torch.equal(old[k],new[k]) and old[k].data_ptr()!=new[k].data_ptr()


def test_control_update_bitwise_replays_previous_low_lr_without_random_draws():
    field,opt,data=setup();fields,opts=resumed_pair(field,opt.state_dict())
    originals,oldopts=previous_pair(field,opt.state_dict());rng=torch.get_rng_state().clone()
    for model,optim in ((fields["control"],opts["control"]),(originals["low_lr"],oldopts["low_lr"])):
        optim.zero_grad(set_to_none=True);cached_edm_loss(model,*data).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step()
    for a,b in zip(fields["control"].parameters(),originals["low_lr"].parameters()):assert torch.equal(a,b)
    assert torch.equal(rng,torch.get_rng_state())
