import torch
from experiments.response_full_models_20260907_convex_head import capture_features,residual_targets,solve_head,replaced_head,body_hash
from experiments.response_full_models_20260907_rb_training import cached_edm_loss
from tests.unit.test_response_full_models_20260907_rb_training import tiny_field


def test_edm_weight_cancellation_is_exact_same_data_objective_and_head_gradient():
    torch.manual_seed(815);field=tiny_field().eval();y=torch.randn(11,4,dtype=torch.float64)
    sigma=torch.logspace(-2,1,11,dtype=torch.float64);target=torch.randn_like(y)
    features=capture_features(field,y,sigma);transformed=residual_targets(y,sigma,target)
    linear=field.output[-1](features);loss=(linear-transformed).square().mean()
    actual=cached_edm_loss(field,y,sigma,target)
    torch.testing.assert_close(loss,actual,rtol=1e-13,atol=1e-13)
    a=torch.autograd.grad(loss,tuple(field.output[-1].parameters()),retain_graph=True)
    b=torch.autograd.grad(actual,tuple(field.output[-1].parameters()))
    for x,z in zip(a,b):torch.testing.assert_close(x,z,rtol=1e-12,atol=1e-12)


def test_svd_cutoff_and_stationarity_on_rank_deficient_design():
    torch.manual_seed(816);x=torch.randn(32,3,dtype=torch.float64);x=torch.cat([x,x[:,:1]],dim=1)
    y=torch.randn(32,2,dtype=torch.float64)
    coeff,design,spectrum,residual,receipt=solve_head(x,y)
    assert receipt["rank"]==4 and receipt["columns"]==5
    assert receipt["rcond"]==torch.finfo(torch.float64).eps*32
    assert receipt["normalized_stationarity"]<1e-13
    torch.testing.assert_close(residual,design@coeff-y)


def test_replacement_changes_only_final_affine_head_and_rounding_is_explicit():
    torch.manual_seed(817);field=tiny_field().eval();old=body_hash(field)
    coefficient=torch.randn(9,4,dtype=torch.float64)
    exact=replaced_head(field,coefficient);rounded=replaced_head(field,coefficient,True)
    assert body_hash(exact)==body_hash(rounded)==old
    assert torch.equal(exact.output[-1].weight,coefficient[:-1].T)
    assert torch.equal(rounded.output[-1].weight,coefficient[:-1].T.float().double())
    assert not torch.equal(field.output[-1].weight,exact.output[-1].weight)


def test_captured_features_replay_actual_final_linear_input_batch_independently():
    torch.manual_seed(818);field=tiny_field().eval();y=torch.randn(9,4,dtype=torch.float64);s=torch.ones(9,dtype=y.dtype)*.2
    a=capture_features(field,y,s,batch_size=2);b=capture_features(field,y,s,batch_size=9)
    torch.testing.assert_close(a,b,rtol=1e-13,atol=1e-13)
    den=1+s.square();expected=y/den[:,None]+s[:,None]/den[:,None].sqrt()*field.output[-1](a)
    from models.training import diffusion_denoiser_output
    torch.testing.assert_close(expected,diffusion_denoiser_output(field,y,s,preconditioning="edm",data_std=1.))


def test_convex_loader_preserves_float64_head_bits(tmp_path):
    from experiments.response_full_models_20260907_convex_head_loader import load_convex_head
    field=tiny_field().double();values=torch.arange(field.output[-1].weight.numel(),dtype=torch.float64).reshape_as(field.output[-1].weight)
    with torch.no_grad():field.output[-1].weight.copy_(values/17+1e-10)
    source={"schema":"ve_convex_head_refit_v1","architecture":field.config.to_dict(),"models":{"refit":field.state_dict()}}
    path=tmp_path/"state.pt";torch.save(source,path)
    loaded=load_convex_head(path,"refit")
    for k,v in field.state_dict().items():assert torch.equal(v,loaded.state_dict()[k])
    assert not torch.equal(loaded.output[-1].weight,loaded.output[-1].weight.float().double())
