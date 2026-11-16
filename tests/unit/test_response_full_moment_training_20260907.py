import copy
from dataclasses import asdict
import math

import numpy as np
import pytest
import torch

import experiments.response_full_moment_training_20260907 as trainer
from experiments.response_full_training_20260907 import objective, posterior, tensor_sha
from models.neural_fields import NeuralFieldConfig, ScaleConditionedNeuralField


def source_payload():
    torch.manual_seed(127)
    arch = NeuralFieldConfig(ambient_dim=3, hidden_dim=8, depth=2,
        condition_dim=6, fourier_features=3, zero_init_output=False)
    model = ScaleConditionedNeuralField(arch)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        objective(model, torch.randn(7,3), torch.randn(7,3), torch.full((7,),.2), "posterior").backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
    return dict(schema="response_full_complex_v1", architecture=asdict(arch), step=3,
        config=dict(device="cpu", dataset="e6_exp_pca", train_count=24,
                    selection_count=8, test_count=8, lambda_min=.01, lambda_max=1.),
        models={"posterior":trainer.cpu_state(model)},
        optimizers={"posterior":copy.deepcopy(opt.state_dict())},
        generator_state=torch.Generator().manual_seed(987).get_state())


def sample():
    rng = torch.Generator().manual_seed(82)
    return torch.randn(7,3,generator=rng,dtype=torch.float64), \
        torch.randn(7,3,generator=rng,dtype=torch.float64), \
        torch.tensor([.01,.03,.05,.1,.3,.7,1.],dtype=torch.float64)


def direct_residual(field, x, z, lam):
    y = (x+lam[:,None]*z).detach().requires_grad_(True)
    b = posterior(field,y,lam,"posterior")
    trace = sum(torch.autograd.grad(b[:,i].sum(),y,create_graph=True,retain_graph=True)[0][:,i]
                for i in range(y.shape[1]))
    residual = trace+((b-y)/lam[:,None]).square().sum(-1)-z.square().sum(-1)
    return residual, trace


def test_stable_residual_exact_trace_and_parameter_gradient_match_direct_ad():
    fields,_ = trainer.paired_state(source_payload(),"cpu")
    model=fields["control"]; x,z,lam=sample()
    stable,response=trainer.moment_residual(model,x,z,lam)
    direct,trace=direct_residual(model,x,z,lam)
    torch.testing.assert_close(stable,direct,rtol=1e-11,atol=1e-10)
    torch.testing.assert_close(response,trace,rtol=1e-11,atol=1e-11)
    params=tuple(model.parameters())
    a=torch.autograd.grad(stable.square().mean()/3,params,retain_graph=True)
    b=torch.autograd.grad(direct.square().mean()/3,params,retain_graph=True)
    for left,right in zip(a,b):
        torch.testing.assert_close(left,right,rtol=2e-10,atol=1e-7)
    detached=direct-trace+trace.detach()
    wrong=torch.autograd.grad(detached.square().mean()/3,params)
    assert max(float((left-right).abs().max()) for left,right in zip(b,wrong)) > .01


def test_source_float32_draw_order_and_generator_restoration_are_exact():
    payload=source_payload()
    first=trainer.restored_generator(payload,"cpu")
    second=trainer.restored_generator(payload,"cpu")
    train=torch.arange(72,dtype=torch.float32).reshape(24,3)/71
    x,z,lam,ids=trainer.draw_batch(train,first,7)
    expected_ids=torch.randint(len(train),(7,),generator=second)
    u=torch.rand(7,generator=second)
    expected_lam=(math.log(.01)+u*math.log(100.)).exp()
    expected_z=torch.randn((7,3),generator=second)
    assert torch.equal(ids,expected_ids)
    assert torch.equal(x,train[expected_ids].double())
    assert torch.equal(z,expected_z.double()) and torch.equal(lam,expected_lam.double())
    assert torch.equal(first.get_state(),second.get_state())
    assert not torch.equal(first.get_state(),payload["generator_state"])
    with pytest.raises(ValueError): trainer.draw_batch(train.double(),first,7)
    with pytest.raises(ValueError): trainer.restored_generator(payload,"cuda:0")


def test_paired_last_and_full_adam_states_are_promoted_without_aliasing():
    source=source_payload(); frozen=copy.deepcopy(source)
    models,optimizers=trainer.paired_state(source,"cpu")
    for a,b in zip(models["control"].parameters(),models["moment"].parameters()):
        assert a.dtype==b.dtype==torch.float64 and a.data_ptr()!=b.data_ptr()
        assert torch.equal(a,b)
    for name in trainer.ARMS:
        state=optimizers[name].state_dict()["state"]
        for key,values in source["optimizers"]["posterior"]["state"].items():
            for field,value in values.items():
                assert state[key][field].dtype==torch.float64
                assert torch.equal(state[key][field],value.double())
    for left,right in zip(optimizers["control"].state.values(),optimizers["moment"].state.values()):
        for name in left: assert left[name].data_ptr()!=right[name].data_ptr()
    trainer.training_step(models["control"],optimizers["control"],*sample(),gamma=0.,
        regularizer_batch_size=5,regularizer_microbatch_size=2)
    assert all(float(s["step"])==4 for s in optimizers["control"].state.values())
    assert all(float(s["step"])==3 for s in optimizers["moment"].state.values())
    for key,value in source["models"]["posterior"].items():
        assert torch.equal(value,frozen["models"]["posterior"][key])
    for key,state in source["optimizers"]["posterior"]["state"].items():
        for name,value in state.items():
            assert torch.equal(value,frozen["optimizers"]["posterior"]["state"][key][name])


@pytest.mark.parametrize("gamma",[0.,.1])
def test_uneven_microbatches_equal_ordinary_total_loss_and_optimizer_step(gamma):
    models,optimizers=trainer.paired_state(source_payload(),"cpu")
    x,z,lam=sample()
    row=trainer.training_step(models["control"],optimizers["control"],x,z,lam,gamma=gamma,
        regularizer_batch_size=5,regularizer_microbatch_size=2)
    ref=models["moment"]; opt=optimizers["moment"]
    opt.zero_grad(set_to_none=True)
    fm=objective(ref,x,z,lam,"posterior")
    reg=trainer.moment_loss(ref,x[:5],z[:5],lam[:5])
    loss=fm+gamma*reg
    loss.backward(); norm=torch.nn.utils.clip_grad_norm_(ref.parameters(),1.); opt.step()
    assert row["moment"]==pytest.approx(float(reg),rel=1e-12)
    assert row["total"]==pytest.approx(float(loss),rel=1e-12)
    assert row["preclip_gradient_norm"]==pytest.approx(float(norm),rel=1e-12)
    for left,right in zip(models["control"].parameters(),ref.parameters()):
        torch.testing.assert_close(left,right,rtol=1e-12,atol=1e-13)
    for a,b in zip(optimizers["control"].state.values(),opt.state.values()):
        for name in a: torch.testing.assert_close(a[name],b[name],rtol=1e-11,atol=1e-13)


def test_three_step_cpu_smoke_preserves_sources_and_fixed_parameter_average(tmp_path,monkeypatch):
    payload=source_payload()
    rng=np.random.default_rng(88)
    splits=[rng.standard_normal((count,3)).astype(np.float32) for count in (24,8,8)]
    raw=torch.from_numpy(splits[0]); mean=raw.mean(0)
    rms=float((raw-mean).square().mean().sqrt())
    payload.update(normalization_mean=mean,normalization_scale=rms)
    fixture=dict(mean=mean,rms=rms,split_hashes={key:tensor_sha(torch.from_numpy(value))
        for key,value in zip(("train","selection","test"),splits)})
    source=tmp_path/"source.pt"; torch.save(payload,source); torch.save(fixture,tmp_path/"fixture.pt")
    before=trainer.sha(source)
    monkeypatch.setattr(trainer,"make_splits",lambda *args,**kwargs:tuple(splits))
    cfg=trainer.Config(source=str(source),output=str(tmp_path/"run"),device="cpu",steps=3,
        batch_size=7,regularizer_batch_size=5,regularizer_microbatch_size=2,average_steps=2)
    models,optimizers=trainer.paired_state(payload,"cpu")
    generator=trainer.restored_generator(payload,"cpu")
    expected={arm:[] for arm in trainer.ARMS}
    for step in range(3):
        x,z,lam,_=trainer.draw_batch((raw-mean)/rms,generator,7)
        for arm in trainer.ARMS:
            trainer.training_step(models[arm],optimizers[arm],x,z,lam,
                gamma=.1 if arm=="moment" else 0.,regularizer_batch_size=5,regularizer_microbatch_size=2)
            if step>=1: expected[arm].append(trainer.cpu_state(models[arm]))
    trainer.run(cfg)
    final=torch.load(tmp_path/"run/final.pt",weights_only=False)
    initial=torch.load(tmp_path/"run/initial.pt",weights_only=False)
    assert trainer.sha(source)==before
    assert final["step"]==6 and final["average_count"]==2 and final["continuation_step"]==3
    assert torch.equal(final["generator_state"],generator.get_state())
    for arm in trainer.ARMS:
        for name,value in models[arm].state_dict().items():
            torch.testing.assert_close(final["models"][arm][name],value,rtol=1e-12,atol=1e-13)
            assert torch.equal(initial["models"][arm][name],payload["models"]["posterior"][name].double())
        for name,_ in models[arm].named_parameters():
            target=(expected[arm][0][name]+expected[arm][1][name])/2
            torch.testing.assert_close(final["averages"][arm][name],target,rtol=1e-12,atol=1e-13)
        for name,_ in models[arm].named_buffers():
            assert torch.equal(final["averages"][arm][name],payload["models"]["posterior"][name].double())
    with pytest.raises(FileExistsError): trainer.run(cfg)


@pytest.mark.parametrize("kwargs",[{"average_steps":2049},{"regularizer_batch_size":1025},
    {"gamma":-1.},{"regularizer_microbatch_size":0}])
def test_invalid_protocol_configuration_rejected(kwargs):
    with pytest.raises(ValueError): trainer.Config(source="unused",output="unused",**kwargs).validate()
