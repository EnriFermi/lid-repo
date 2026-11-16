from dataclasses import replace
import copy

import pytest
import torch

from experiments.fm_mechanism_pilot import PilotConfig,train_pilot,canonical_posterior
from experiments.fm_mechanism_fast_trace import fast_value_and_trace
from experiments.fm_scheduler_cause_training import (ANCHOR,ARM_SCHEDULE,arm_multiplier,
    arm_posterior,evaluation_copy,load_cause_checkpoint,original_loss,parser,run)
from models.neural_fields import NeuralFieldConfig,ScaleConditionedNeuralField


@pytest.mark.parametrize("coordinate",("data","native"))
def test_anchor_pairs_have_equal_functions_and_parameter_gradients(coordinate):
    torch.manual_seed(42)
    field=ScaleConditionedNeuralField(NeuralFieldConfig(ambient_dim=3,hidden_dim=8,depth=1,zero_init_output=False)).double()
    field.pilot_output_coordinate=coordinate
    x=torch.randn(4,3,dtype=torch.float64)
    for a,b in (("rect","vp_low"),("rect_high","vp")):
        left,right=arm_posterior(field,x,ANCHOR,a),arm_posterior(field,x,ANCHOR,b)
        torch.testing.assert_close(left,right,rtol=1e-12,atol=1e-12)
        gl=torch.autograd.grad(left.square().mean(),tuple(field.parameters()))
        gr=torch.autograd.grad(right.square().mean(),tuple(field.parameters()))
        for u,v in zip(gl,gr):torch.testing.assert_close(u,v,rtol=1e-10,atol=1e-12)


@pytest.mark.parametrize("arm",tuple(ARM_SCHEDULE))
@pytest.mark.parametrize("coordinate",("data","native"))
@pytest.mark.parametrize("normalization",("layernorm","none"))
def test_folded_copy_exact_value_trace_and_source_unchanged(arm,coordinate,normalization):
    torch.manual_seed(43)
    field=ScaleConditionedNeuralField(NeuralFieldConfig(ambient_dim=3,hidden_dim=8,depth=1,zero_init_output=False)).double()
    field.pilot_output_coordinate=coordinate
    if normalization=="none":field.output[0]=torch.nn.Identity()
    saved=copy.deepcopy(field.state_dict())
    points=torch.randn(3,3,dtype=torch.float64,requires_grad=True)
    value=arm_posterior(field,points,.1,arm)
    trace=sum(torch.autograd.grad(value[:,i].sum(),points,retain_graph=True)[0][:,i] for i in range(3))
    folded=evaluation_copy(field,arm)
    fv,ft=fast_value_and_trace(folded,points,.1,ARM_SCHEDULE[arm])
    torch.testing.assert_close(fv,value,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(ft,trace,rtol=1e-11,atol=1e-12)
    assert all(torch.equal(v,saved[k]) for k,v in field.state_dict().items())


def test_full_small_training_replays_unchanged_pilot_and_preserves_rng(tmp_path):
    cfg=PilotConfig(ambient_dim=8,intrinsic_dim=2,hidden_dim=16,depth=1,batch_size=8,
        steps=2,validation_count=16,log_interval=2,checkpoints=(0,2),device="cpu",data_law="uniform")
    train_pilot(tmp_path/"source",cfg)
    args=parser().parse_args(["--source",str(tmp_path/"source/checkpoint_step_000000.pt"),
        "--output",str(tmp_path/"new"),"--device","cpu","--steps","4","--decay-step","2",
        "--tail-average","2","--tail-probes","2","--query-count","16","--heldout-count","16",
        "--probe-count","8","--log-interval","2","--checkpoints","0","2","4"])
    result=run(args)
    assert result["source_unchanged"]
    baseline=torch.load(tmp_path/"source/checkpoint_step_000002.pt",weights_only=False)
    for arm,schedule in (("rect","rectified_linear"),("vp","vp_trigonometric")):
        field,payload=load_cause_checkpoint(tmp_path/f"new/{arm}/checkpoint_step_000002.pt")
        assert all(torch.equal(v,baseline["models"][schedule][k]) for k,v in field.state_dict().items())
        assert torch.equal(payload["training_generator_state"],baseline["training_generator_state"])
        final=torch.load(tmp_path/f"new/{arm}/checkpoint_step_000004.pt",weights_only=False)
        assert all(int(v["step"])==4 for v in final["optimizer"]["state"].values())
        avg=torch.load(tmp_path/f"new/{arm}_tailavg/checkpoint_step_000004.pt",weights_only=False)
        assert avg["optimizer"] is None and avg["inference_only"]
    probes=torch.load(tmp_path/"new/tail_probes.pt",weights_only=False)
    assert len(probes["observations"])==2*4*3


def test_unknown_arm_and_foreign_checkpoint_rejected(tmp_path):
    with pytest.raises(ValueError):arm_multiplier("bad")
    path=tmp_path/"wrong.pt"
    torch.save({"schema_version":1},path)
    with pytest.raises(ValueError,match="anchored-path"):load_cause_checkpoint(path)


def test_gaussian_law_override_preserves_dimension_basis_and_initial_weights(tmp_path):
    cfg=PilotConfig(ambient_dim=8,intrinsic_dim=2,hidden_dim=16,depth=1,batch_size=8,
        steps=2,validation_count=16,log_interval=2,checkpoints=(0,2),device="cpu",data_law="uniform")
    train_pilot(tmp_path/"source",cfg)
    args=parser().parse_args(["--source",str(tmp_path/"source/checkpoint_step_000000.pt"),
        "--output",str(tmp_path/"gaussian"),"--device","cpu","--data-law","gaussian",
        "--steps","2","--decay-step","1","--tail-average","1","--tail-probes","1",
        "--query-count","16","--heldout-count","16","--probe-count","8","--checkpoints","0","2"])
    run(args)
    source_fixture=torch.load(tmp_path/"source/data.pt",weights_only=False)
    generated=torch.load(tmp_path/"gaussian/data.pt",weights_only=False)
    assert generated["data_law"]=="gaussian"
    assert torch.equal(source_fixture["tangent_basis"],generated["tangent_basis"])
    source=torch.load(tmp_path/"source/checkpoint_step_000000.pt",weights_only=False)
    generated_payload=torch.load(tmp_path/"gaussian/rect/checkpoint_step_000000.pt",weights_only=False)
    assert generated_payload["config"]["data_law"]=="gaussian"
    assert all(torch.equal(v,source["models"]["rectified_linear"][k]) for k,v in generated_payload["model"].items())


def test_no_final_layernorm_preserves_common_initial_tensors_and_loader(tmp_path):
    cfg=PilotConfig(ambient_dim=8,intrinsic_dim=2,hidden_dim=16,depth=1,batch_size=8,
        steps=2,validation_count=16,log_interval=2,checkpoints=(0,2),device="cpu",data_law="uniform")
    train_pilot(tmp_path/"source",cfg)
    args=parser().parse_args(["--source",str(tmp_path/"source/checkpoint_step_000000.pt"),
        "--output",str(tmp_path/"no_norm"),"--device","cpu","--output-normalization","none",
        "--steps","2","--decay-step","1","--tail-average","1","--tail-probes","1",
        "--query-count","16","--heldout-count","16","--probe-count","8","--checkpoints","0","2"])
    run(args)
    original=torch.load(tmp_path/"source/checkpoint_step_000000.pt",weights_only=False)["models"]["rectified_linear"]
    field,payload=load_cause_checkpoint(tmp_path/"no_norm/rect/checkpoint_step_000000.pt")
    assert isinstance(field.output[0],torch.nn.Identity)
    assert isinstance(field.blocks[0].norm,torch.nn.LayerNorm)
    assert set(original)-set(field.state_dict())=={"output.0.weight","output.0.bias"}
    assert all(torch.equal(v,original[k]) for k,v in field.state_dict().items())


def test_total_native_compensation_formula_loss_gradients_and_loader(tmp_path):
    from experiments.fm_mechanism_diagnostics import alpha_at
    torch.manual_seed(17)
    field=ScaleConditionedNeuralField(NeuralFieldConfig(ambient_dim=3,hidden_dim=8,depth=1,zero_init_output=False)).double()
    field.output[0]=torch.nn.Identity()
    field.pilot_output_coordinate="native"
    clean,noise=torch.randn(2,5,3,dtype=torch.float64)
    scales=torch.tensor([.01,.1,ANCHOR,.6,1.],dtype=torch.float64)
    for arm in ARM_SCHEDULE:
        total=arm_multiplier(arm)*alpha_at(scales,ARM_SCHEDULE[arm])
        direct=field(total[:,None]*(clean+scales[:,None]*noise),scales.log())/total[:,None]
        prediction=arm_posterior(field,clean+scales[:,None]*noise,scales,arm)
        torch.testing.assert_close(prediction,direct,rtol=1e-12,atol=1e-12)
        native_loss=original_loss(field,clean,noise,scales,arm)
        direct_loss=((direct-clean).square().mean(-1)/scales.square()).mean()
        left=torch.autograd.grad(native_loss,tuple(field.parameters()))
        right=torch.autograd.grad(direct_loss,tuple(field.parameters()))
        for a,b in zip(left,right):torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)
    cfg=PilotConfig(ambient_dim=8,intrinsic_dim=2,hidden_dim=16,depth=1,batch_size=8,
        steps=2,validation_count=16,log_interval=2,checkpoints=(0,2),device="cpu",data_law="uniform")
    train_pilot(tmp_path/"source",cfg)
    args=parser().parse_args(["--source",str(tmp_path/"source/checkpoint_step_000000.pt"),
        "--output",str(tmp_path/"native"),"--device","cpu","--output-normalization","none",
        "--output-coordinate","native","--steps","2","--decay-step","1",
        "--tail-average","1","--tail-probes","1","--query-count","16",
        "--heldout-count","16","--probe-count","8","--checkpoints","0","2"])
    run(args)
    for arm in ARM_SCHEDULE:
        restored,payload=load_cause_checkpoint(tmp_path/f"native/{arm}/checkpoint_step_000002.pt")
        assert restored.pilot_output_coordinate=="native"
        assert payload["config"]["output_coordinate"]=="native"
        assert isinstance(restored.output[0],torch.nn.Identity)
    original=torch.load(tmp_path/"source/checkpoint_step_000000.pt",weights_only=False)["models"]["rectified_linear"]
    initial=torch.load(tmp_path/"native/rect/checkpoint_step_000000.pt",weights_only=False)["model"]
    assert all(torch.equal(v,original[k]) for k,v in initial.items())


@pytest.mark.parametrize("arm",tuple(ARM_SCHEDULE))
def test_crossed_output_direct_formula_parameter_gradient_and_exact_trace(arm):
    from experiments.fm_mechanism_diagnostics import alpha_at
    from experiments.fm_scheduler_cause_training import CROSSED_OUTPUT_ARM,evaluated_posterior,exact_arrays
    torch.manual_seed(117)
    field=ScaleConditionedNeuralField(NeuralFieldConfig(ambient_dim=3,hidden_dim=8,depth=1,zero_init_output=False)).double()
    field.pilot_output_coordinate="native"
    field.cause_native_output_profile="crossed"
    clean,noise=torch.randn(2,5,3,dtype=torch.float64)
    scales=torch.tensor([.01,.1,ANCHOR,.6,1.],dtype=torch.float64)
    inputs=clean+scales[:,None]*noise
    input_alpha=arm_multiplier(arm)*alpha_at(scales,ARM_SCHEDULE[arm])
    other=CROSSED_OUTPUT_ARM[arm]
    output_alpha=arm_multiplier(other)*alpha_at(scales,ARM_SCHEDULE[other])
    direct=field(input_alpha[:,None]*inputs,scales.log())/output_alpha[:,None]
    value=arm_posterior(field,inputs,scales,arm)
    torch.testing.assert_close(value,direct,rtol=1e-12,atol=1e-12)
    left=torch.autograd.grad(original_loss(field,clean,noise,scales,arm),tuple(field.parameters()))
    right=torch.autograd.grad(((direct-clean).square().mean(-1)/scales.square()).mean(),tuple(field.parameters()))
    for a,b in zip(left,right):torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)
    points=inputs[:3].detach().requires_grad_()
    value=arm_posterior(field,points,.1,arm)
    trace=sum(torch.autograd.grad(value[:,i].sum(),points,retain_graph=True)[0][:,i] for i in range(3))
    frozen=evaluation_copy(field,arm)
    result,exact,correction=exact_arrays(frozen,points,.1,ARM_SCHEDULE[arm])
    torch.testing.assert_close(result,value,rtol=1e-11,atol=1e-12)
    torch.testing.assert_close(exact,trace,rtol=1e-11,atol=1e-12)
    torch.testing.assert_close(evaluated_posterior(frozen,points,.1,ARM_SCHEDULE[arm]),value,rtol=1e-11,atol=1e-12)
    torch.testing.assert_close(correction,(value-points).square().sum(-1)/.1**2)
    matched=copy.deepcopy(field);matched.cause_native_output_profile="matched"
    a=arm_posterior(field,inputs,ANCHOR,arm)
    b=arm_posterior(matched,inputs,ANCHOR,arm)
    torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    ga=torch.autograd.grad(a.square().mean(),tuple(field.parameters()))
    gb=torch.autograd.grad(b.square().mean(),tuple(matched.parameters()))
    for u,v in zip(ga,gb):torch.testing.assert_close(u,v,rtol=1e-10,atol=1e-12)


def test_crossed_profile_small_training_loader_and_preserved_defaults(tmp_path):
    cfg=PilotConfig(ambient_dim=8,intrinsic_dim=2,hidden_dim=16,depth=1,batch_size=8,
        steps=2,validation_count=16,log_interval=2,checkpoints=(0,2),device="cpu",data_law="uniform")
    train_pilot(tmp_path/"source",cfg)
    args=parser().parse_args(["--source",str(tmp_path/"source/checkpoint_step_000000.pt"),
        "--output",str(tmp_path/"crossed"),"--device","cpu","--output-coordinate","native",
        "--native-output-profile","crossed","--steps","2","--decay-step","1",
        "--tail-average","1","--tail-probes","1","--query-count","16",
        "--heldout-count","16","--probe-count","8","--checkpoints","0","2"])
    run(args)
    for arm in ARM_SCHEDULE:
        field,payload=load_cause_checkpoint(tmp_path/f"crossed/{arm}_tailavg/checkpoint_step_000002.pt")
        assert payload["config"]["native_output_profile"]=="crossed"
        assert field.cause_native_output_profile=="crossed"
        assert field.pilot_output_coordinate=="native"
        assert isinstance(field.output[0],torch.nn.LayerNorm)
