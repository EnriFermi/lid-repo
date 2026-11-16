import torch
from experiments.response_full_models_20260907_fixed_objective_lr import resumed_pair
from experiments.response_full_models_20260907_rb_training import cached_edm_loss
from experiments.response_full_models_20260907_fixed_objective import resumed_pair as original_pair
from tests.unit.test_response_full_models_20260907_rb_training import tiny_field


def setup():
    torch.manual_seed(451);field=tiny_field().float()
    y=torch.randn(7,4);target=torch.randn_like(y);sigma=torch.linspace(.01,1,7)
    opt=torch.optim.AdamW(field.parameters(),lr=.001,weight_decay=.0001)
    cached_edm_loss(field,y,sigma,target).backward();opt.step()
    return field,opt,(y,sigma,target)


def test_both_rates_inherit_all_adam_state_without_reset_or_aliasing():
    field,opt,_=setup();fields,opts=resumed_pair(field,opt.state_dict())
    for arm in fields:
        assert opts[arm].param_groups[0]["lr"]==(.001 if arm=="original_lr" else 2e-5)
        for old,new in zip(opt.state.values(),opts[arm].state.values()):
            for key in ("step","exp_avg","exp_avg_sq"):
                assert torch.equal(old[key],new[key]) and old[key].data_ptr()!=new[key].data_ptr()
        for old,new in zip(field.parameters(),fields[arm].parameters()):assert torch.equal(old,new)


def test_control_step_bitwise_matches_frozen_parent_full_gradient_update():
    field,opt,data=setup();fields,opts=resumed_pair(field,opt.state_dict())
    originals,oldopts=original_pair(field,opt.state_dict())
    for model,optim in ((fields["original_lr"],opts["original_lr"]),(originals["full_gradient"],oldopts["full_gradient"])):
        optim.zero_grad(set_to_none=True);cached_edm_loss(model,*data).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step()
    for a,b in zip(fields["original_lr"].parameters(),originals["full_gradient"].parameters()):assert torch.equal(a,b)


def test_first_paired_update_has_identical_moments_and_lr_scaled_displacement():
    field,opt,data=setup();fields,opts=resumed_pair(field,opt.state_dict())
    for arm,model in fields.items():
        opts[arm].zero_grad(set_to_none=True);cached_edm_loss(model,*data).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opts[arm].step()
    for a,b in zip(opts["original_lr"].state.values(),opts["low_lr"].state.values()):
        for key in ("step","exp_avg","exp_avg_sq"):assert torch.equal(a[key],b[key])
    for base,hi,lo in zip(field.parameters(),fields["original_lr"].parameters(),fields["low_lr"].parameters()):
        torch.testing.assert_close(lo,base+.02*(hi-base),atol=1e-7,rtol=1e-7)
