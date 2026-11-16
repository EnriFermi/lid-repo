import copy
import torch
from experiments.response_full_models_20260907_fixed_objective import resumed_pair
from experiments.response_full_models_20260907_rb_training import cached_edm_loss
from tests.unit.test_response_full_models_20260907_rb_training import tiny_field


def test_enumerated_uniform_singleton_raw_gradients_equal_fixed_objective_gradient():
    torch.manual_seed(42);field=tiny_field();y=torch.randn(5,4,dtype=torch.float64)
    target=torch.randn_like(y);sigma=torch.linspace(.01,2,5,dtype=torch.float64)
    loss=cached_edm_loss(field,y,sigma,target)
    full=torch.autograd.grad(loss,tuple(field.parameters()))
    mean=[torch.zeros_like(g) for g in full]
    for i in range(5):
        one=cached_edm_loss(field,y[i:i+1],sigma[i:i+1],target[i:i+1])
        grad=torch.autograd.grad(one,tuple(field.parameters()))
        for total,g in zip(mean,grad):total.add_(g/5)
    for a,b in zip(full,mean):torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-11)


def test_resumed_adam_state_and_weights_identical_but_independent():
    torch.manual_seed(43);field=tiny_field().float()
    opt=torch.optim.AdamW(field.parameters(),lr=.001,weight_decay=.0001)
    y=torch.randn(6,4);sigma=torch.full((6,),.1);target=torch.randn_like(y)
    cached_edm_loss(field,y,sigma,target).backward();opt.step()
    fields,opts=resumed_pair(field,opt.state_dict())
    for arm in fields:
        for p,q in zip(field.parameters(),fields[arm].parameters()):
            assert torch.equal(p,q) and p.data_ptr()!=q.data_ptr()
        assert opts[arm].param_groups[0]["lr"]==.001
        for old,new in zip(opt.state.values(),opts[arm].state.values()):
            for key in ("step","exp_avg","exp_avg_sq"):
                assert torch.equal(old[key],new[key]) and old[key].data_ptr()!=new[key].data_ptr()
    first=next(iter(opts["full_gradient"].state.values()))["exp_avg"]
    second=next(iter(opts["stochastic_gradient"].state.values()))["exp_avg"]
    before=second.clone();first.add_(1);assert torch.equal(second,before)


def test_stochastic_rows_are_replayable_uniform_with_replacement_draws():
    a=torch.Generator().manual_seed(20260910);b=torch.Generator().manual_seed(20260910)
    first=torch.randint(8192,(512,256),generator=a)
    second=torch.stack([torch.randint(8192,(256,),generator=b) for _ in range(512)])
    assert torch.equal(first,second) and torch.equal(a.get_state(),b.get_state())
    assert first.min()>=0 and first.max()<8192 and len(torch.unique(first[0]))<256


def test_dense_lags_detect_alternation_after_linear_trend_removal():
    import numpy as np
    from experiments.response_full_models_20260907_fixed_objective_dense import lag_summary
    steps=np.arange(128);signal=(.04*steps+(-1.)**steps)[:,None,None]
    rows,spectrum=lag_summary(signal,steps)
    lag1=next(r for r in rows if r["series"]=="linear_detrended" and r["lag"]==1)
    assert lag1["normalized_inner_product"]<-.999
    assert spectrum["peak_period_updates"]==2
    assert spectrum["peak_fraction_of_nonzero_rfft_power"]>.999
