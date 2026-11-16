"""Paired noise-head continuation covering the interior of the training shell.

From one immutable final model and Adam state, compare original full Gaussian
noise with a half-batch mixture of normal-noise amplitude .25 and1. All X,
global-span noise, physical scales, optimizer settings and budgets are shared.
The optimal canonical posterior is unchanged because these globally absent
coordinates are independent of data. Gradient amplitudes are not unchanged:
this tests an ordinary resampling intervention, not pure coverage at fixed
parameter-gradient geometry. No oracle labels or LID selection are used.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from experiments.response_full_models_20260907_shell import normal_blocks
from experiments.response_full_models_20260907_span import correction_components, projected_trace, sha256
from experiments.response_full_models_20260907_ve import SCALES
from experiments.response_full_models_20260907_vp import load_field
from experiments.response_full_training_20260907 import objective, posterior, readout, tensor_sha
from experiments.ve_known_lid_recoverability_campaign import make_splits, DATASET_SPECS


def mix_normal_noise(z, basis, amplitude=.25):
    """Exactly half the batch has contracted normal noise; other rows unchanged."""
    if not 0 < amplitude <= 1 or len(z)%2:
        raise ValueError("Require positive amplitude and even batch size")
    if amplitude == 1:
        return z
    span = (z@basis)@basis.T
    changed = span+amplitude*(z-span)
    mask = (torch.arange(len(z),device=z.device)%2 == 0)[:,None]
    return torch.where(mask,changed,z)


def clone_continuations(payload, device, lr):
    fields, opts = {}, {}
    for name in ["control", "inner_shell"]:
        field = load_field(payload,"noise","last",device).float()
        opt = torch.optim.AdamW(field.parameters(),lr=lr,weight_decay=1e-4)
        opt.load_state_dict(copy.deepcopy(payload["optimizers"]["noise"]))
        for group in opt.param_groups:
            group["lr"] = lr
        fields[name],opts[name] = field,opt
    for a,b in zip(fields["control"].parameters(),fields["inner_shell"].parameters()):
        assert torch.equal(a,b) and a.data_ptr()!=b.data_ptr()
    for a,b in zip(opts["control"].state.values(),opts["inner_shell"].state.values()):
        for key in ["exp_avg","exp_avg_sq","step"]:
            assert torch.equal(a[key],b[key]) and a[key].data_ptr()!=b[key].data_ptr()
    return fields,opts


def save(path, fields, opts, rng, config, architecture, mean, rms, step, averages=None):
    torch.save({"schema":"response_full_shell_continuation_v1", "config":config,
        "architecture":architecture,"step":step,"models":{name:{k:v.detach().cpu().clone()
            for k,v in model.state_dict().items()} for name,model in fields.items()},
        "optimizers":{name:opt.state_dict() for name,opt in opts.items()},
        "generator_state":rng.get_state(),"normalization_mean":mean,
        "normalization_scale":rms,"averages":averages},path)


def run_law(args, law):
    source=args.root/f"vp_{law}_paired"
    out=args.output/law;out.mkdir(parents=True,exist_ok=False)
    checkpoint=source/"final.pt";before=sha256(checkpoint)
    payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
    fixture=torch.load(source/"fixture.pt",map_location="cpu",weights_only=False)
    span_fixture=torch.load(args.root/f"vp_{law}_span/span_and_queries.pt",map_location="cpu",weights_only=False)
    config=payload["config"]
    if args.average_steps>args.steps or args.steps<1:raise ValueError("Invalid budget")
    train_raw,_,_=make_splits(config["dataset"],train_size=config["train_count"],
        validation_size=config["selection_count"],test_size=config["test_count"],data_seed=0)
    assert tensor_sha(torch.from_numpy(train_raw))==fixture["split_hashes"]["train"]
    mean,rms=fixture["mean"],fixture["rms"]
    train=((torch.from_numpy(train_raw)-mean)/rms).to(args.device)
    basis=span_fixture["basis"].float().to(args.device)
    fields,opts=clone_continuations(payload,args.device,args.learning_rate)
    rng=torch.Generator(device=args.device);rng.set_state(payload["generator_state"].cpu())
    eval_rng=torch.Generator(device=args.device).manual_seed(config["seed"]+2)
    validation=fixture["validation"].to(args.device)
    eval_noise=torch.randn(validation.shape,generator=eval_rng,device=args.device)
    manifest={"source_checkpoint":str(checkpoint),"source_checkpoint_sha256":before,
        "source_sha256":sha256(__file__),"source_parent_step":payload["step"],
        "source_adapter_sha256":sha256("experiments/response_full_training_20260907.py"),
        "steps":args.steps,"average_steps":args.average_steps,"learning_rate":args.learning_rate,
        "optimizer":"Restored original per-parameter AdamW moments and step counters; clip1, weight decay1e-4",
        "draws":"Restored source CUDA generator; same original randint, uniform-lambda, Gaussian-noise ordering; no intervention-only RNG consumption",
        "mixture":f"Even batch rows have normal-noise amplitude {args.amplitude}; odd rows amplitude1; ordinary clean-X scalar objective unchanged",
        "mixture_amplitude":args.amplitude,
        "gradient_caveat":"Normal target and input amplitude change, and original-loss normal gradient contribution changes; not a pure coverage effect at fixed gradients",
        "selection":f"Fixed final step and last{args.average_steps} parameter average, no oracle/LID selection",
        "fixture_sha256":sha256(source/"fixture.pt"),"queries":len(fixture["queries"]),
        "dataset":config["dataset"],"initial_parameters_and_optimizer_states_exactly_paired":True}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    derived_config={**config,"continuation_steps":args.steps,"continuation_learning_rate":args.learning_rate,
        "inner_normal_amplitude":args.amplitude,"continuation_source":str(checkpoint)}
    save(out/"step_000000.pt",fields,opts,rng,derived_config,payload["architecture"],mean,rms,0)
    average={name:{k:torch.zeros_like(v,dtype=torch.float64,device="cpu")
        for k,v in model.state_dict().items()} for name,model in fields.items()}
    history=[];start=time.monotonic()
    for step in range(1,args.steps+1):
        index=torch.randint(len(train),(config["batch_size"],),generator=rng,device=args.device)
        x=train[index]
        u=torch.rand(config["batch_size"],generator=rng,device=args.device)
        lam=(math.log(config["lambda_min"])+u*math.log(config["lambda_max"]/config["lambda_min"])).exp()
        z=torch.randn(x.shape,generator=rng,device=args.device)
        mixed=mix_normal_noise(z,basis,args.amplitude)
        row={"step":step}
        for name,model in fields.items():
            model.train();opts[name].zero_grad(set_to_none=True)
            loss=objective(model,x,z if name=="control" else mixed,lam,"noise")
            if not torch.isfinite(loss):raise FloatingPointError(name)
            loss.backward();gnorm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opts[name].step()
            row[name+"_loss"]=float(loss.detach());row[name+"_grad_norm"]=float(gnorm)
            if step>args.steps-args.average_steps:
                for key,value in model.state_dict().items():
                    average[name][key].add_(value.detach().cpu().double(),alpha=1/args.average_steps)
        if step%256==0 or step==args.steps:
            history.append(row)
            print(json.dumps({"law":law,"elapsed_seconds":time.monotonic()-start,**row}),flush=True)
            (out/"status.json").write_text(json.dumps({"status":"training","step":step,"elapsed_seconds":time.monotonic()-start})+"\n")
        if step in [args.steps//2,args.steps]:
            save(out/f"step_{step:06d}.pt",fields,opts,rng,derived_config,payload["architecture"],mean,rms,step)
    averages={name:{k:v.float() for k,v in state.items()} for name,state in average.items()}
    save(out/"final.pt",fields,opts,rng,derived_config,payload["architecture"],mean,rms,args.steps,averages)
    (out/"history.json").write_text(json.dumps(history,indent=2)+"\n")
    queries=fixture["queries"].double().to(args.device)
    center=span_fixture["center"].double().to(args.device)
    basis=basis.double()
    # Use original float64 fitted basis for exact derivative/projection algebra.
    basis=span_fixture["basis"].double().to(args.device)
    normal=torch.linalg.qr(basis,mode="complete").Q[:,basis.shape[1]:]
    shell_rng=torch.Generator().manual_seed(20261009)
    shell_noise=torch.randn(queries.shape,generator=shell_rng,dtype=torch.float64).to(args.device)
    shell_noise=(shell_noise@normal)@normal.T
    rows,shell_rows,shell_raw=[],[],{}
    for name,field in fields.items():
        for mode in ["last","average"]:
            model=copy.deepcopy(field).double().eval()
            if mode=="average":model.load_state_dict(averages[name])
            values={}
            for scale in SCALES:
                with torch.no_grad():
                    b,r,c=readout(model,queries,scale,"noise")
                    risk=float(objective(model,validation.double(),eval_noise.double(),
                        torch.full((len(validation),),scale,device=args.device,dtype=torch.float64),"noise"))
                _,span_r=projected_trace(lambda q:posterior(model,q,scale,"noise"),queries,basis,16)
                terms=correction_components(b,queries,center,basis,scale)
                projected=center+((b-center)@basis)@basis.T
                full=r+c;projected_full=span_r+terms["correction_projected"]
                tensors={"posterior":b,"posterior_projected":projected,"response":r,"response_span":span_r,
                    "response_offspan":r-span_r,"full":full,"full_projected":projected_full,**terms}
                for key,value in tensors.items():values.setdefault(key,[]).append(value.detach().cpu().numpy())
                d=DATASET_SPECS[config["dataset"]].true_lid
                rows.append({"law":law,"arm":name,"mode":mode,"lambda":scale,
                    "full_d_mae":float((full-d).abs().mean()),"response_d_mae":float((r-d).abs().mean()),
                    "projected_full_d_mae":float((projected_full-d).abs().mean()),
                    "normal_response_mae":float((r-span_r).abs().mean()),
                    "normal_response_mean":float((r-span_r).mean()),"correction_mean":float(c.mean()),
                    "original_gaussian_noisy_validation_risk":risk})
            np.savez_compressed(out/f"{name}_{mode}_arrays.npz",queries=queries.cpu().numpy(),query_raw=fixture["query_raw"],
                query_ids=fixture["query_ids"],scales=np.asarray(SCALES),**{k:np.stack(v) for k,v in values.items()})
            if mode=="average":
                for scale in [.1,.316227766]:
                    for rho in [0.,1.]:
                        data=normal_blocks(lambda q:posterior(model,q,scale,"noise"),queries+rho*scale*shell_noise,
                            normal,center,scale,16)
                        for key,value in data.items():
                            shell_raw[f"{name}_lambda{scale}_rho{rho}_{key}"]=value.detach().cpu().numpy()
                        shell_rows.append({"law":law,"arm":name,"lambda":scale,"rho":rho,
                            "normal_trace_mean":float(data["normal_trace"].mean()),
                            "normal_trace_mae":float(data["normal_trace"].abs().mean()),
                            "normal_value_energy_over_lambda_squared":float(data["normal_value_energy_over_lambda_squared"].mean()),
                            "normal_jacobian_frobenius_squared_per_dimension":float(data["normal_jacobian_frobenius_squared_per_dimension"].mean())})
    np.savez_compressed(out/"shell_arrays.npz",query_ids=fixture["query_ids"],**shell_raw)
    for filename,data in [("metrics",rows),("shell_metrics",shell_rows)]:
        (out/f"{filename}.json").write_text(json.dumps(data,indent=2)+"\n")
        with (out/f"{filename}.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    assert sha256(checkpoint)==before and sha256(__file__)==manifest["source_sha256"]
    (out/"status.json").write_text(json.dumps({"status":"complete","steps":args.steps,
        "elapsed_seconds":time.monotonic()-start,"source_checkpoint_unchanged":True})+"\n")
    return {"law":law,"output":str(out),"elapsed_seconds":time.monotonic()-start}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("artifacts/response_full_20260907"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--laws",nargs="+",default=["spiral","exp"])
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--steps",type=int,default=2560)
    p.add_argument("--average-steps",type=int,default=512)
    p.add_argument("--learning-rate",type=float,default=2e-5)
    p.add_argument("--amplitude",type=float,default=.25)
    args=p.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    for law in args.laws:print(json.dumps(run_law(args,law)),flush=True)


if __name__=="__main__":main()
