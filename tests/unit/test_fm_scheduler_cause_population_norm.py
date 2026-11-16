import copy
import math

import pytest
import torch

from experiments.fm_mechanism_pilot import PilotConfig,train_pilot
from experiments.fm_scheduler_cause_population_norm import (PopulationRMSNorm,
    install_population_norm,population_grid,population_value_and_trace)
from experiments.fm_scheduler_cause_training import (ARM_SCHEDULE,arm_posterior,
    evaluation_copy,load_cause_checkpoint,parser,run)
from models.neural_fields import NeuralFieldConfig,ScaleConditionedNeuralField


def normalization_config():
    return {"log_grid":population_grid(.01,1.,16).tolist(),"decay":.9}


def test_forward_is_actual_population_model_and_gradcheck_preserves_affine():
    original=torch.nn.LayerNorm(5).double()
    with torch.no_grad():
        original.weight.copy_(torch.tensor([.5,1.,1.5,2.,2.5],dtype=torch.float64))
        original.bias.copy_(torch.arange(5,dtype=torch.float64)/10)
    norm=PopulationRMSNorm(original,log_grid=[-5.,0.]).double()
    norm.running_variance.copy_(torch.tensor([2.,4.],dtype=torch.float64))
    assert torch.equal(norm.weight,original.weight) and torch.equal(norm.bias,original.bias)
    x=torch.randn(3,5,dtype=torch.float64,requires_grad=True)
    logs=torch.tensor([-4.,-2.,-.1],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x,l:norm.forward_at(x,l),(x,logs),eps=1e-6,atol=1e-5)
    y=norm.forward_at(x,logs)
    expected=(x-x.mean(-1,keepdim=True))/(norm.variance_at(logs)+norm.eps).sqrt()[:,None]*norm.weight+norm.bias
    torch.testing.assert_close(y,expected)
    # Its centered output scales with x: this is not sample-LN with stopped grad.
    torch.testing.assert_close(norm.forward_at(2*x,logs)-norm.bias,2*(y-norm.bias))


def test_variance_buffers_do_not_change_until_explicit_post_step_commit():
    norm=PopulationRMSNorm(torch.nn.LayerNorm(5),log_grid=[-5.,0.],decay=.9)
    norm.train();norm._log_scales=torch.tensor([-3.,-2.,-1.])
    x=torch.randn(3,5,requires_grad=True)*3
    before=norm.running_variance.clone()
    loss=norm(x).square().mean()
    assert torch.equal(before,norm.running_variance)
    loss.backward()
    assert torch.equal(before,norm.running_variance)
    assert all(not t.requires_grad for t in norm._pending)
    norm.commit_statistics()
    assert not torch.equal(before,norm.running_variance)
    assert int(norm.updates)==1
    torch.testing.assert_close(norm.observation_count.sum(),torch.tensor(3.))
    with pytest.raises(RuntimeError,match="pending"):norm.commit_statistics()


@pytest.mark.parametrize("arm",tuple(ARM_SCHEDULE))
def test_exact_spatial_trace_matches_autograd_and_eval_batch_is_separable(arm):
    torch.manual_seed(56)
    field=ScaleConditionedNeuralField(NeuralFieldConfig(ambient_dim=3,hidden_dim=8,depth=2,zero_init_output=False)).double()
    install_population_norm(field,normalization_config())
    field.output[0].running_variance.copy_(torch.linspace(.2,2.,len(field.output[0].log_grid)))
    field.eval()
    x=torch.randn(4,3,dtype=torch.float64,requires_grad=True)
    scale=.316227766
    values=arm_posterior(field,x,scale,arm)
    trace=sum(torch.autograd.grad(values[:,i].sum(),x,retain_graph=True)[0][:,i] for i in range(3))
    frozen=evaluation_copy(field,arm)
    actual_values,actual_trace=population_value_and_trace(frozen,x,scale,ARM_SCHEDULE[arm])
    torch.testing.assert_close(actual_values,values,rtol=1e-11,atol=1e-12)
    torch.testing.assert_close(actual_trace,trace,rtol=1e-10,atol=1e-12)
    buffers={k:v.clone() for k,v in field.named_buffers()}
    singles=torch.cat([arm_posterior(field,p,scale,arm) for p in x.split(1)])
    torch.testing.assert_close(singles,values,rtol=1e-11,atol=1e-12)
    assert all(torch.equal(v,buffers[k]) for k,v in field.named_buffers())


def test_grid_includes_prespecified_anchors_and_interpolation_is_continuous_positive():
    grid=population_grid(.01,1.,16)
    assert len(grid)==18
    for anchor in (.1,.316227766,1.):
        assert bool((grid==math.log(anchor)).any())
    norm=PopulationRMSNorm(torch.nn.LayerNorm(5).double(),log_grid=grid)
    norm.running_variance.copy_(torch.linspace(.1,2.,len(grid)))
    for center in grid[1:-1]:
        values=norm.variance_at(torch.stack((center-1e-9,center,center+1e-9)))
        assert bool((values>0).all())
        assert float((values-values[1]).abs().max())<1e-7


def test_training_loader_calibration_and_average_recalibration(tmp_path):
    cfg=PilotConfig(ambient_dim=8,intrinsic_dim=2,hidden_dim=16,depth=1,batch_size=8,
        steps=2,validation_count=16,log_interval=2,checkpoints=(0,2),device="cpu",data_law="uniform")
    train_pilot(tmp_path/"source",cfg)
    args=parser().parse_args(["--source",str(tmp_path/"source/checkpoint_step_000000.pt"),
        "--output",str(tmp_path/"population"),"--device","cpu","--output-normalization","population",
        "--population-initial-calibration-per-knot","8","--population-average-calibration-per-knot","16",
        "--steps","2","--decay-step","1","--tail-average","1","--tail-probes","1",
        "--query-count","16","--heldout-count","16","--probe-count","8","--checkpoints","0","2"])
    result=run(args)
    assert result["source_unchanged"]
    old=torch.load(tmp_path/"source/checkpoint_step_000002.pt",weights_only=False)
    for arm in ARM_SCHEDULE:
        field,current=load_cause_checkpoint(tmp_path/f"population/{arm}/checkpoint_step_000002.pt")
        assert isinstance(field.output[0],PopulationRMSNorm)
        assert int(field.output[0].updates)==2
        assert torch.equal(current["training_generator_state"],old["training_generator_state"])
        averaged,payload=load_cause_checkpoint(tmp_path/f"population/{arm}_tailavg/checkpoint_step_000002.pt")
        assert int(averaged.output[0].updates)==0
        assert payload["population_average_stats_recalibration"]["examples_per_knot"]==16
        assert payload["population_average_stats_recalibration"]["total_examples"]==16*18
        expected_count=torch.tensor(16*18,dtype=averaged.output[0].observation_count.dtype)
        torch.testing.assert_close(averaged.output[0].observation_count.sum(),expected_count)
        assert payload["optimizer"] is None and payload["inference_only"]
