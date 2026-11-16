"""Paired original-loss training: anchored path shape x input amplitude.

Four input-coordinate ablations share initialization, original clean-X labels,
physical scales and all random draws. Custom checkpoints deliberately cannot
be read as unmodified pilot checkpoints: the input amplitude must be respected.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from experiments.fm_mechanism_condition_refit import make_oracle
from experiments.fm_mechanism_diagnostics import alpha_at
from experiments.fm_mechanism_fast_trace import fast_value_and_trace
from experiments.fm_mechanism_gradient_transfer import sample
from experiments.fm_mechanism_pilot import PilotConfig, canonical_posterior, load_pilot_checkpoint, make_fixture, state_digest
from experiments.fm_mechanism_regional_transfer import observed_regions
from experiments.fm_scheduler_cause_population_norm import (PopulationRMSNorm,
    calibrate_population, install_population_norm, population_grid, population_value_and_trace)
from models.neural_fields import NeuralFieldConfig, ScaleConditionedNeuralField


SCHEMA = "fm_scheduler_cause_v1"
ARM_SCHEDULE = {"rect": "rectified_linear", "vp": "vp_trigonometric",
                "vp_low": "vp_trigonometric", "rect_high": "rectified_linear"}
ANCHOR = .316227766
EVAL_SCALES = (.1, ANCHOR, 1.)
CROSSED_OUTPUT_ARM = {"rect":"vp_low","vp_low":"rect","rect_high":"vp","vp":"rect_high"}


def arm_multiplier(arm, anchor=ANCHOR):
    if arm not in ARM_SCHEDULE:
        raise ValueError("Unknown anchored-path arm")
    c = math.sqrt(1+anchor**2)/(1+anchor)
    return {"rect": 1., "vp": 1., "vp_low": c, "rect_high": 1/c}[arm]


def arm_posterior(field, inputs, scales, arm, anchor=ANCHOR):
    multiplier = arm_multiplier(arm, anchor)
    if getattr(field,"cause_native_output_profile","matched")=="crossed":
        if getattr(field,"pilot_output_coordinate","data")!="native":
            raise ValueError("Crossed output requires native output coordinate")
        raw=canonical_posterior(field,inputs*multiplier,scales,ARM_SCHEDULE[arm],output_coordinate="data")
        output_arm=CROSSED_OUTPUT_ARM[arm]
        lam=torch.as_tensor(scales,device=inputs.device,dtype=inputs.dtype)
        if lam.ndim==0:lam=lam.expand(len(inputs))
        denominator=arm_multiplier(output_arm,anchor)*alpha_at(lam,ARM_SCHEDULE[output_arm])
        return raw/denominator[:,None]
    prediction = canonical_posterior(field, inputs*multiplier, scales, ARM_SCHEDULE[arm])
    # canonical_posterior handles base alpha; native output also compensates
    # the anchored arm's CONSTANT multiplier, hence TOTAL native amplitude.
    if getattr(field, "pilot_output_coordinate", "data") == "native":
        prediction = prediction/multiplier
    return prediction


def draw_batch(config, fixture, generator, device):
    """Match the original pilot's private draw stream, including operation order."""
    count, d, D = config["batch_size"], config["intrinsic_dim"], config["ambient_dim"]
    law = config["data_law"]
    if law == "gaussian":
        latent = torch.randn(count, d, device=device, generator=generator)
    elif law == "uniform":
        latent = math.sqrt(3)*(2*torch.rand(count, d, device=device, generator=generator)-1)
    elif law == "circle":
        latent = 2*math.pi*torch.rand(count, 1, device=device, generator=generator)
    else:
        raise ValueError("Unsupported data law")
    if law == "circle":
        clean = math.sqrt(2)*torch.cat((latent.cos(), latent.sin()), dim=1)
    else:
        clean = (float(fixture["latent_std"])*latent) @ fixture["tangent_basis"].to(device).T
    noise = torch.randn(count, D, device=device, generator=generator)
    uniform = torch.rand(count, device=device, generator=generator)
    scales = (math.log(config["lambda_min"])+uniform*math.log(config["lambda_max"]/config["lambda_min"])).exp()
    return clean, noise, scales


def original_loss(field, clean, noise, scales, arm, anchor=ANCHOR):
    prediction = arm_posterior(field, clean+scales[:,None]*noise, scales, arm, anchor)
    return ((prediction-clean).square().mean(-1)/scales.square()).mean()


def evaluation_copy(field, arm, anchor=ANCHOR):
    """Fold constants into a COPY; crossed output REQUIRES our eval helpers.

    Shared canonical_posterior/fast_value_and_trace know only one schedule;
    evaluated_posterior/exact_arrays additionally apply the crossed profile.
    """
    result = copy.deepcopy(field).double().eval()
    with torch.no_grad():
        result.input.weight.mul_(arm_multiplier(arm, anchor))
        if getattr(field, "pilot_output_coordinate", "data") == "native":
            output_arm=CROSSED_OUTPUT_ARM[arm] if getattr(field,"cause_native_output_profile","matched")=="crossed" else arm
            result.output[-1].weight.div_(arm_multiplier(output_arm, anchor))
            result.output[-1].bias.div_(arm_multiplier(output_arm, anchor))
            if output_arm!=arm:result.cause_trace_output_schedule=ARM_SCHEDULE[output_arm]
    return result


def evaluation_output_factor(field,scales,schedule,reference):
    output_schedule=getattr(field,"cause_trace_output_schedule",None)
    if output_schedule is None:return None
    lam=torch.as_tensor(scales,device=reference.device,dtype=reference.dtype)
    if lam.ndim==0:lam=lam.expand(len(reference))
    return alpha_at(lam,schedule)/alpha_at(lam,output_schedule)


def evaluated_posterior(field,points,scales,schedule):
    """Canonical posterior of an evaluation_copy, including crossed output."""
    value=canonical_posterior(field,points,scales,schedule)
    factor=evaluation_output_factor(field,scales,schedule,points)
    return value if factor is None else value*factor[:,None]


def load_cause_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SCHEMA:
        raise ValueError("Expected an anchored-path causal-training checkpoint")
    field = ScaleConditionedNeuralField(NeuralFieldConfig.from_mapping(payload["architecture"]))
    if payload["config"].get("output_normalization", "layernorm") == "none":
        field.output[0] = torch.nn.Identity()
    elif payload["config"].get("output_normalization") == "population":
        install_population_norm(field,payload["config"]["population_norm"])
    field.load_state_dict(payload["model"], strict=True)
    field.pilot_output_coordinate = payload["config"].get("output_coordinate", "data")
    field.cause_native_output_profile=payload["config"].get("native_output_profile","matched")
    return field.to(device).eval(), payload


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def exact_arrays(field, points, scale, schedule):
    tracer=population_value_and_trace if isinstance(field.output[0],PopulationRMSNorm) else fast_value_and_trace
    parts = [tracer(field, p, scale, schedule) for p in points.split(32)]
    values = torch.cat([p[0] for p in parts])
    response = torch.cat([p[1] for p in parts])
    factor=evaluation_output_factor(field,scale,schedule,points)
    if factor is not None:
        values=values*factor[:,None]
        response=response*factor
    correction = (values-points).square().sum(-1)/scale**2
    return values, response, correction


@torch.no_grad()
def evaluate(field, arm, config, fixture, evaluation_data, oracle, tangent_energy, anchor=ANCHOR):
    evaluated = evaluation_copy(field, arm, anchor)
    queries, clean, noise = (evaluation_data[k] for k in ("queries","clean","noise"))
    rows, arrays = [], {}
    schedule = ARM_SCHEDULE[arm]
    for scale in EVAL_SCALES:
        values, response, correction = exact_arrays(evaluated, queries, scale, schedule)
        truth = oracle(queries, scale)
        noisy = clean+scale*noise
        population=evaluated.output[0] if isinstance(evaluated.output[0],PopulationRMSNorm) else None
        predictions_list,variance_ratios=[],[]
        if population is not None:population.capture_statistics=True
        for p in noisy.split(256):
            predictions_list.append(evaluated_posterior(evaluated,p,scale,schedule))
            if population is not None:
                logs,variance=population.take_observations()
                variance_ratios.append((variance+population.eps)/(population.variance_at(logs)+population.eps))
        if population is not None:population.capture_statistics=False
        predictions=torch.cat(predictions_list)
        ratio_values=torch.cat(variance_ratios) if variance_ratios else None
        noisy_truth = oracle(noisy, scale)
        masks = {"all": torch.ones(len(queries), dtype=torch.bool, device=queries.device)}
        if fixture["data_law"] == "uniform":
            masks.update(observed_regions(queries, fixture, scale))
        noisy_risk = float((predictions-clean).square().mean()/scale**2)
        excess_risk = float((predictions-noisy_truth.mean).square().mean()/scale**2)
        noisy_tangent = float(tangent_energy(predictions-noisy_truth.mean, clean).mean()/scale**2)
        if fixture["data_law"] in {"uniform","gaussian"}:
            basis = torch.linalg.qr(fixture["tangent_basis"].to(noisy)).Q
            noisy_radial = (noisy@basis)@basis.T
        else:
            noisy_radial = noisy
        noisy_radial_error = float(((predictions-noisy_truth.mean)*noisy_radial).sum()/noisy_radial.square().sum())
        for region, mask in masks.items():
            if int(mask.sum()) < 8:
                continue
            error = values[mask]-truth.mean[mask]
            q = queries[mask]
            tangent = tangent_energy(error,q)
            total = error.square().sum(-1)
            full = response[mask]+correction[mask]
            rows.append({"arm":arm,"lambda":scale,"region":region,"count":int(mask.sum()),
                "heldout_clean_x_risk":noisy_risk,"heldout_oracle_excess_risk":excess_risk,
                "heldout_tangent_error_over_lambda2":noisy_tangent,
                "heldout_radial_gain_error":noisy_radial_error,
                "population_variance_ratio_mean":float(ratio_values.mean()) if ratio_values is not None else float("nan"),
                "population_variance_ratio_se":float(ratio_values.std()/math.sqrt(len(ratio_values))) if ratio_values is not None else float("nan"),
                "full_mae_d":float((full-config["intrinsic_dim"]).abs().mean()),
                "full_mae_oracle":float((full-truth.full[mask]).abs().mean()),
                "response_mean":float(response[mask].mean()),"correction_mean":float(correction[mask].mean()),
                "response_mae_d":float((response[mask]-config["intrinsic_dim"]).abs().mean()),
                "response_mae_oracle":float((response[mask]-truth.response[mask]).abs().mean()),
                "radial_gain_error":float((error*q).sum()/q.square().sum().clamp_min(1e-30)),
                "clean_tangent_error_over_lambda2":float(tangent.mean()/scale**2),
                "clean_normal_error_over_lambda2":float((total-tangent).mean()/scale**2)})
        arrays[scale] = {"values":values.cpu(),"response":response.cpu(),"correction":correction.cpu(),
            "oracle_values":truth.mean.cpu(),"oracle_response":truth.response.cpu(),
            "oracle_correction":truth.correction.cpu(),"oracle_full":truth.full.cpu(),
            "heldout_values":predictions.cpu(),"heldout_oracle_values":noisy_truth.mean.cpu()}
    return rows, arrays


def run(args):
    run_started=time.monotonic()
    if args.output_coordinate == "native" and args.output_normalization == "population":
        raise ValueError("Population normalization plus native output is not a specified intervention")
    if args.native_output_profile=="crossed" and args.output_coordinate!="native":
        raise ValueError("Crossed output profile requires --output-coordinate native")
    if not 0 < args.decay_step < args.steps or args.tail_average <= 0 or args.tail_average > args.steps:
        raise ValueError("Require 0<decay_step<steps and valid positive tail-average length")
    if args.tail_probes <= 0 or args.tail_probes > args.steps:
        raise ValueError("Invalid tail-probe length")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    payload = torch.load(args.source, map_location="cpu", weights_only=False)
    if payload["step"] != 0:
        raise ValueError("This fresh paired experiment must start at the preserved step0")
    config = copy.deepcopy(payload["config"])
    if config.get("input_coordinate","native") != "native" or config.get("output_coordinate","data") != "data":
        raise ValueError("Require native-input, data-output posterior source")
    if config.get("head_preconditioner","elementwise") != "elementwise":
        raise ValueError("Require ordinary AdamW source")
    args.output.mkdir(parents=True, exist_ok=False)
    fixture_source = args.source.parent/"data.pt"
    fixture = torch.load(fixture_source, map_location="cpu", weights_only=False)
    fixture.setdefault("data_law",config["data_law"])
    if args.data_law is not None and args.data_law != config["data_law"]:
        config["data_law"] = args.data_law
        fixture_config = PilotConfig(**{k:v for k,v in config.items() if k in PilotConfig.__dataclass_fields__})
        changed_fixture = make_fixture(fixture_config)
        assert torch.equal(changed_fixture["tangent_basis"],fixture["tangent_basis"])
        fixture = changed_fixture
    torch.save(fixture,args.output/"data.pt")
    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    template,_ = load_pilot_checkpoint(args.source,"rectified_linear",args.device)
    config["output_coordinate"] = args.output_coordinate
    template.pilot_output_coordinate = args.output_coordinate
    config["native_output_profile"]=args.native_output_profile
    template.cause_native_output_profile=args.native_output_profile
    if args.output_normalization is not None:
        if args.output_normalization == "none":
            template.output[0] = torch.nn.Identity()
        elif args.output_normalization == "population":
            config["population_norm"]={"log_grid":population_grid(config["lambda_min"],config["lambda_max"],
                args.population_bins,EVAL_SCALES).tolist(),"decay":args.population_decay,
                "initial_calibration_per_knot":args.population_initial_calibration_per_knot,
                "average_calibration_per_knot":args.population_average_calibration_per_knot}
            install_population_norm(template,config["population_norm"])
        elif not isinstance(template.output[0],torch.nn.LayerNorm):
            raise ValueError("Cannot introduce a new LayerNorm from a non-normalized source")
        config["output_normalization"] = args.output_normalization
    fields = {arm:copy.deepcopy(template).train() for arm in ARM_SCHEDULE}
    optimizers = {arm:torch.optim.AdamW(f.parameters(),lr=args.initial_lr,weight_decay=config["weight_decay"])
                  for arm,f in fields.items()}
    assert len({state_digest(f.state_dict()) for f in fields.values()}) == 1
    assert all(a.data_ptr()!=b.data_ptr() for a,b in zip(fields["rect"].parameters(),fields["vp"].parameters()))
    generator = torch.Generator(device=args.device)
    generator.set_state(payload["training_generator_state"])
    queries,_ = sample(fixture,args.query_count,20260905+7102,args.device)
    clean,noise = sample(fixture,args.heldout_count,20260905+7101,args.device)
    eval_data = {"queries":queries,"clean":clean,"noise":noise}
    torch.save({k:v.cpu() for k,v in eval_data.items()},args.output/"evaluation_data.pt")
    oracle,tangent_energy = make_oracle(config,fixture,args.device)
    config.update(device=args.device,steps=args.steps,learning_rate=args.initial_lr)
    initial_calibration={}
    if args.output_normalization=="population":
        for arm,field in fields.items():
            initial_calibration[arm]=calibrate_population(field,arm,config,fixture,
                per_knot=args.population_initial_calibration_per_knot,seed=config["seed"]+9701,
                device=args.device,anchor=args.anchor,predict=arm_posterior,draw_batch=draw_batch)
        (args.output/"population_calibration_initial.json").write_text(json.dumps(initial_calibration,indent=2)+"\n")
    manifest = {"schema_version":SCHEMA,"source":str(args.source),"source_sha256":source_hash,
        "source_fixture_sha256":hashlib.sha256(fixture_source.read_bytes()).hexdigest(),
        "effective_fixture_sha256":hashlib.sha256((args.output/"data.pt").read_bytes()).hexdigest(),
        "data_law_override":args.data_law,"effective_config":config,
        "output_normalization_override":args.output_normalization,
        "output_coordinate_override":args.output_coordinate,
        "native_output_profile":args.native_output_profile,
        "crossed_output_arms":CROSSED_OUTPUT_ARM if args.native_output_profile=="crossed" else None,
        "output_coordinate_contract":"native divides by TOTAL designated-output-arm multiplier*base_alpha (same input arm if matched, crossed_output_arms mapping if crossed); data leaves network output unchanged",
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "population_norm_source_sha256":hashlib.sha256(Path(__file__).with_name("fm_scheduler_cause_population_norm.py").read_bytes()).hexdigest(),
        "source_config":payload["config"],"architecture":payload["architecture"],
        "arms":{a:{"base_schedule":s,"input_amplitude_multiplier":arm_multiplier(a,args.anchor)} for a,s in ARM_SCHEDULE.items()},
        "anchor":args.anchor,"evaluation_scales":EVAL_SCALES,
        "steps":args.steps,"decay_step":args.decay_step,"initial_lr":args.initial_lr,"final_lr":args.final_lr,
        "checkpoints":args.checkpoints,"query_count":args.query_count,"heldout_count":args.heldout_count,
        "tail_average_length":args.tail_average,"tail_probe_length":args.tail_probes,
        "initial_state_hash":state_digest(template.state_dict()),"paired_draws":True,
        "initial_state_hashes_after_calibration":{a:state_digest(f.state_dict()) for a,f in fields.items()},
        "population_normalization":config.get("population_norm"),
        "population_stats_protocol":"Previous-step buffers in forward/backward; detached hidden-variance EMA after optimizer.step; no evaluation updates; averaged-weight stats freshly recalibrated",
        "loss":"Original clean-X mean((posterior-X)^2/(D lambda^2)); no oracle training targets",
        "optimizer":"Full ordinary AdamW incl first/second moments, decay and original global norm clipping",
        "factorial":"shape(rect/VP) x input amplitude at anchor(low/high); matching anchor input pairs rect/vp_low and rect_high/vp",
        "path_scope":"Anchored paths are input-coordinate ablations, not named globally endpoint-normalized schedules",
        "frozen_before_results":True,"torch_version":torch.__version__}
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    checkpoints = set(args.checkpoints)|{0,args.steps}
    history, eval_rows, tail_values = [],[],[]
    averages = {}
    started = run_started

    def snapshot(step,averaged=False):
        for arm in ARM_SCHEDULE:
            directory = args.output/(arm+("_tailavg" if averaged else ""))
            directory.mkdir(exist_ok=True)
            field_state = averages[arm] if averaged else fields[arm].state_dict()
            evaluated = copy.deepcopy(fields[arm])
            if averaged:evaluated.load_state_dict(field_state)
            recalibration=None
            if averaged and isinstance(evaluated.output[0],PopulationRMSNorm):
                recalibration=calibrate_population(evaluated,arm,config,fixture,
                    per_knot=args.population_average_calibration_per_knot,seed=config["seed"]+9702,
                    device=args.device,anchor=args.anchor,predict=arm_posterior,draw_batch=draw_batch)
                (directory/"population_calibration.json").write_text(json.dumps(recalibration,indent=2)+"\n")
            state = {k:v.detach().cpu().clone() for k,v in evaluated.state_dict().items()}
            optimizer_state = None if averaged else copy.deepcopy(optimizers[arm].state_dict())
            if not averaged and step:
                assert all(float(v["step"])==step for v in optimizer_state["state"].values())
            current = {"schema_version":SCHEMA,"step":step,"arm":arm,"anchor":args.anchor,
                "input_amplitude_multiplier":arm_multiplier(arm,args.anchor),"base_schedule":ARM_SCHEDULE[arm],
                "config":config,"architecture":payload["architecture"],"model":state,"optimizer":optimizer_state,
                "training_generator_state":generator.get_state(),"source_sha256":source_hash,
                "population_average_stats_recalibration":recalibration,
                "averaged":averaged,"inference_only":averaged,"average_steps":list(range(args.steps-args.tail_average+1,args.steps+1)) if averaged else None,
                "examples_seen":step*config["batch_size"]}
            torch.save(current,directory/f"checkpoint_step_{step:06d}.pt")
            rows,arrays = evaluate(evaluated,arm,config,fixture,eval_data,oracle,tangent_energy,args.anchor)
            eval_rows.extend({"step":step,"averaged":averaged,**r} for r in rows)
            torch.save(arrays,directory/f"evaluation_step_{step:06d}.pt")
        write_csv(args.output/"evaluation.csv",eval_rows)

    snapshot(0)
    for step in range(1,args.steps+1):
        if step == args.decay_step+1:
            for optimizer in optimizers.values():
                for group in optimizer.param_groups:
                    group["lr"] = args.final_lr
        clean_batch,noise_batch,scales = draw_batch(config,fixture,generator,args.device)
        for arm,field in fields.items():
            optimizer = optimizers[arm]
            optimizer.zero_grad(set_to_none=True)
            loss = original_loss(field,clean_batch,noise_batch,scales,arm,args.anchor)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError((step,arm,"nonfinite original loss"))
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(field.parameters(),config["gradient_clip"])
            optimizer.step()
            if isinstance(field.output[0],PopulationRMSNorm):field.output[0].commit_statistics()
            history.append({"step":step,"arm":arm,"loss":float(loss),"gradient_norm":float(norm),
                            "learning_rate":optimizer.param_groups[0]["lr"]})
            if step > args.steps-args.tail_average:
                with torch.no_grad():
                    count = step-(args.steps-args.tail_average)
                    if arm not in averages:
                        averages[arm] = {k:v.detach().clone() for k,v in field.state_dict().items()}
                    else:
                        for name,value in field.state_dict().items():
                            if value.is_floating_point():
                                averages[arm][name].add_((value-averages[arm][name])/count)
            if step > args.steps-args.tail_probes:
                with torch.no_grad():
                    frozen = evaluation_copy(field,arm,args.anchor)
                    for scale in EVAL_SCALES:
                        value,response,correction = exact_arrays(frozen,queries[:args.probe_count],scale,ARM_SCHEDULE[arm])
                        tail_values.append({"step":step,"arm":arm,"lambda":scale,"values":value.cpu(),
                            "response":response.cpu(),"correction":correction.cpu()})
        if step in checkpoints:
            snapshot(step)
        if step%args.log_interval == 0 or step==args.steps:
            write_csv(args.output/"training_history.csv",history)
            status={"step":step,"elapsed_seconds":time.monotonic()-started,"status":"running",
                    "losses":{r["arm"]:r["loss"] for r in history[-len(ARM_SCHEDULE):]}}
            (args.output/"status.json").write_text(json.dumps(status,indent=2)+"\n")
            print(json.dumps(status),flush=True)
    snapshot(args.steps,averaged=True)
    torch.save({"queries":queries[:args.probe_count].cpu(),"observations":tail_values},args.output/"tail_probes.pt")
    assert hashlib.sha256(args.source.read_bytes()).hexdigest()==source_hash
    summary={"status":"complete","elapsed_seconds":time.monotonic()-started,"steps":args.steps,
        "source_unchanged":True,"optimizer_steps_verified":True,
        "final_state_hashes":{a:state_digest(f.state_dict()) for a,f in fields.items()},
        "training_rng_sha256":hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest()}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (args.output/"status.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary),flush=True)
    return summary


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source",type=Path,required=True)
    result.add_argument("--output",type=Path,required=True)
    result.add_argument("--device",default="cuda:1")
    result.add_argument("--data-law",choices=("uniform","gaussian","circle"))
    result.add_argument("--output-normalization",choices=("layernorm","none","population"))
    result.add_argument("--output-coordinate",choices=("data","native"),default="data")
    result.add_argument("--native-output-profile",choices=("matched","crossed"),default="matched")
    result.add_argument("--population-bins",type=int,default=16)
    result.add_argument("--population-decay",type=float,default=.9)
    result.add_argument("--population-initial-calibration-per-knot",type=int,default=2048)
    result.add_argument("--population-average-calibration-per-knot",type=int,default=32768)
    result.add_argument("--anchor",type=float,default=ANCHOR)
    result.add_argument("--steps",type=int,default=9000)
    result.add_argument("--decay-step",type=int,default=8000)
    result.add_argument("--initial-lr",type=float,default=.001)
    result.add_argument("--final-lr",type=float,default=.0001)
    result.add_argument("--tail-average",type=int,default=500)
    result.add_argument("--tail-probes",type=int,default=25)
    result.add_argument("--probe-count",type=int,default=128)
    result.add_argument("--query-count",type=int,default=512)
    result.add_argument("--heldout-count",type=int,default=2048)
    result.add_argument("--log-interval",type=int,default=500)
    result.add_argument("--checkpoints",nargs="+",type=int,default=[0,1000,3000,8000,8500,9000])
    return result


if __name__=="__main__":
    print(json.dumps(run(parser().parse_args())),flush=True)
