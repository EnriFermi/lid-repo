"""Paired learning-rate change for a deterministic fixed-objective gradient.

This resumes the same LAST weights AND Adam state. Both use the identical full
objective, so learning rate is the only intervention.
"""
from __future__ import annotations
import argparse,copy,csv,json,time
from pathlib import Path
import numpy as np
import torch
from experiments.response_full_models_20260907_span import sha256
from experiments.response_full_models_20260907_rb_training import cached_edm_loss
from experiments.response_full_models_20260907_ve_refinement import diagnostics,state_cpu,original_loss,load_refinement_field
from experiments.response_full_models_20260907_ve import SCALES
from experiments.ve_known_lid_recoverability_campaign import make_splits
from models.training import diffusion_denoiser_output

ARMS=("original_lr","low_lr")


def resumed_pair(source,optimizer_state):
    fields={arm:copy.deepcopy(source).float() for arm in ARMS}
    opts={}
    for arm,field in fields.items():
        opt=torch.optim.AdamW(field.parameters(),lr=.001)
        opt.load_state_dict(copy.deepcopy(optimizer_state))
        for group in opt.param_groups:group["lr"]=.001 if arm=="original_lr" else 2e-5
        opts[arm]=opt
    return fields,opts


def save(path,fields,opts,source,step,averages=None):
    torch.save({"schema":"ve_fixed_objective_lr_refinement_v1","step":step,
        "parent_step":source["step"],"architecture":source["architecture"],
        "training_config":source["training_config"],"normalization":source["normalization"],
        "models":{k:state_cpu(v) for k,v in fields.items()},
        "optimizers":{k:v.state_dict() for k,v in opts.items()},"averages":averages},path)


def value_probe(field,queries,sigma=.01):
    model=copy.deepcopy(field).double().eval()
    with torch.no_grad():
        b=diffusion_denoiser_output(model,queries.double(),sigma,preconditioning="edm",data_std=1.)
    return b.cpu().numpy()


def run(args):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True,exist_ok=False)
    source=torch.load(args.source,map_location="cpu",weights_only=False)
    cfg=source["training_config"]
    if source["schema"]!="ve_fixed_objective_refinement_v1" or source["step"]!=512:
        raise ValueError("Require completed fixed-objective LAST checkpoint, not parameter average")
    if args.rows<1 or args.average_steps>args.steps or args.average_steps%8:
        raise ValueError("Invalid fixed objective/budget")
    source_manifest=args.source.with_name("manifest.json")
    source_status=args.source.with_name("status.json")
    if json.loads(source_status.read_text())["status"]!="complete":raise ValueError("Incomplete parent")
    old_manifest=json.loads(source_manifest.read_text())
    inputs=[args.source,args.cache,args.queries,source_manifest,source_status,Path(__file__),
        Path("experiments/response_full_models_20260907_rb_training.py"),
        Path("experiments/response_full_models_20260907_ve_refinement.py"),Path("models/training.py")]
    hashes={str(p):sha256(p) for p in inputs}
    if old_manifest["source_hashes"][str(args.cache)]!=hashes[str(args.cache)]:
        raise ValueError("Cache differs from completed parent")
    cache=torch.load(args.cache,map_location="cpu",weights_only=False)
    n=len(cache["sigma"]);ids=torch.arange(n-args.rows,n)
    tape={k:cache[k][ids].to(args.device) for k in ("observed","sigma","oracle_target")}
    mean=source["normalization"]["mean"].cpu();rms=float(source["normalization"]["scale"])
    torch.testing.assert_close(torch.as_tensor(cache["normalization_mean"]),mean,rtol=0,atol=0)
    assert float(cache["normalization_scale"])==rms
    assert cache["dataset"]=="e8_spaghetti_pca" and cache["fresh_continuous_population"]
    field=load_refinement_field(args.source,"full_gradient","last",args.device).float()
    fields,opts=resumed_pair(field,source["optimizers"]["full_gradient"])
    for arm,opt in opts.items():
        assert all(float(v["step"])==2560 for v in opt.state.values())
        expected_lr=.001 if arm=="original_lr" else 2e-5
        assert all(g["lr"]==expected_lr and g["weight_decay"]==cfg["weight_decay"] for g in opt.param_groups)
    fixture=np.load(args.queries);q=torch.from_numpy(fixture["queries"]).to(args.device)
    _,val_raw,_=make_splits("e8_spaghetti_pca",train_size=100000,validation_size=1000,test_size=1000,data_seed=0)
    validation=((torch.from_numpy(val_raw)-mean)/rms).to(args.device)
    rng_before=torch.get_rng_state().clone()
    cuda_rng_before=torch.cuda.get_rng_state(args.device).clone() if str(args.device).startswith("cuda") else None
    manifest={"source_hashes":hashes,"parent_arm":"full_gradient","parent_mode":"last",
        "optimizer":"Both deep-copy parent LAST Adam moments/counter2560, weight_decay and clip1; only LR.001 versus2e-5 differs",
        "rows":args.rows,"row_start_in_parent_tape":int(ids[0]),"row_stop_exclusive":n,
        "fixed_objective":"Mean original EDM-weighted posterior-target loss over identical last8192 cached observations",
        "steps":args.steps,"full_batch_size":args.rows,
        "sampling":"No resampling: both full8192-row objective at every update",
        "learning_rates":{"original_lr":.001,"low_lr":2e-5},"average_steps":args.average_steps,"snapshot_interval":8,
        "snapshot_scales":[.01,.1,.316227766,1.],"snapshot_queries":64,
        "per_step_values":"Every final128 update, sigma.01 on same first64 queries; exact double forward",
        "comparison":"Same weights/moments/objective/updates/FLOPs; only learning rate differs",
        "scope":"Deterministic LR-controlled finite-step dynamics; not a unique Adam mechanism or asymptotic cycle claim",
        "selection":"Fixed budget/final128 parameter average, no LID selection",
        "source_training_config":"Inherited ancestry; actual rows/batch sizes/updates are this manifest"}
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    np.save(args.output/"objective_row_ids.npy",ids.numpy())
    save(args.output/"step_000000.pt",fields,opts,source,0)
    initial=diagnostics(fields[ARMS[0]],q,SCALES)
    old=np.load(args.source.with_name("full_gradient_last_arrays.npz"))
    for k in ("posterior","response","correction_original"):
        np.testing.assert_allclose(initial[k],old[k],rtol=1e-9,atol=1e-9)
    np.savez_compressed(args.output/"initial_arrays.npz",queries=fixture["queries"],query_ids=fixture["query_ids"],scales=SCALES,**initial)
    averages={a:{k:torch.zeros_like(v,dtype=torch.float64,device="cpu") for k,v in f.state_dict().items()} for a,f in fields.items()}
    snapshots={a:[] for a in ARMS};values={a:[] for a in ARMS};snapshot_steps=[];value_steps=[]
    history=[];fixed_losses=[];start=time.monotonic()
    def fixed_loss_row(step):
        row={"step":step}
        with torch.no_grad():
            for arm,model in fields.items():
                model.eval();row[arm]=float(cached_edm_loss(model,tape["observed"],tape["sigma"],tape["oracle_target"]))
        fixed_losses.append(row)
    fixed_loss_row(0)
    for step in range(1,args.steps+1):
        row={"step":step}
        for arm,model in fields.items():
            y,sigma,target=[tape[k] for k in ("observed","sigma","oracle_target")]
            model.train();opts[arm].zero_grad(set_to_none=True)
            loss=cached_edm_loss(model,y,sigma,target)
            if not torch.isfinite(loss):raise FloatingPointError(arm)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["gradient_clip_norm"])
            opts[arm].step()
            row.update({arm+"_sample_loss":float(loss.detach()),arm+"_gradient_norm":float(norm),
                arm+"_clip_factor":min(1.,float(cfg["gradient_clip_norm"])/(float(norm)+1e-6))})
            if step>args.steps-args.average_steps:
                for k,v in model.state_dict().items():averages[arm][k].add_(v.detach().cpu().double(),alpha=1/args.average_steps)
                values[arm].append(value_probe(model,q[:64]))
        history.append(row)
        if step>args.steps-args.average_steps:
            value_steps.append(step)
            if (step-(args.steps-args.average_steps))%8==0:
                snapshot_steps.append(step)
                for arm,model in fields.items():snapshots[arm].append(diagnostics(model,q[:64],manifest["snapshot_scales"]))
        if step%64==0:
            fixed_loss_row(step)
            print(json.dumps({"elapsed_seconds":time.monotonic()-start,**row,"fixed_losses":fixed_losses[-1]}),flush=True)
            (args.output/"status.json").write_text(json.dumps({"status":"training","step":step})+"\n")
            save(args.output/f"step_{step:06d}.pt",fields,opts,source,step)
    averages={a:{k:v.float() for k,v in state.items()} for a,state in averages.items()}
    save(args.output/"final.pt",fields,opts,source,args.steps,averages)
    assert torch.equal(torch.get_rng_state(),rng_before)
    if cuda_rng_before is not None:assert torch.equal(torch.cuda.get_rng_state(args.device),cuda_rng_before)
    torch.save({"cpu":rng_before,"cuda":cuda_rng_before,"unchanged":True},args.output/"rng_receipt.pt")
    rows=[];temporal_rows=[]
    for arm,model in fields.items():
        np.savez_compressed(args.output/f"{arm}_per_step_values.npz",steps=np.asarray(value_steps),
            posterior=np.stack(values[arm]),queries=fixture["queries"][:64],query_ids=fixture["query_ids"][:64],sigma=.01)
        for mode in ("last","average"):
            evaluated=copy.deepcopy(model)
            if mode=="average":evaluated.load_state_dict(averages[arm])
            data=diagnostics(evaluated,q,SCALES);evaluated.eval()
            with torch.no_grad():
                risk=float(original_loss(evaluated,validation,cfg,torch.Generator(device=args.device).manual_seed(20261209)))
                objective=float(cached_edm_loss(evaluated,tape["observed"],tape["sigma"],tape["oracle_target"]))
            np.savez_compressed(args.output/f"{arm}_{mode}_arrays.npz",queries=fixture["queries"],query_ids=fixture["query_ids"],scales=SCALES,**data)
            for i,s in enumerate(SCALES):rows.append({"arm":arm,"mode":mode,"lambda":s,
                "response_mean":float(data["response"][i].mean()),"correction_mean":float(data["correction_original"][i].mean()),
                "response_d_mae":float(np.abs(data["response"][i]-1).mean()),"full_d_mae":float(np.abs(data["full"][i]-1).mean()),
                "original_edm_validation_risk":risk,"fixed_objective_loss":objective})
        seq=snapshots[arm];data={k:np.stack([item[k] for item in seq]) for k in seq[0]}
        b=data["posterior"].mean(0);r=data["response"].mean(0);scales=np.asarray(manifest["snapshot_scales"])
        c=((b-fixture["queries"][:64][None])**2).sum(-1)/scales[:,None]**2
        variation=((data["posterior"]-b[None])**2).sum(-1).mean(0)/scales[:,None]**2
        np.testing.assert_allclose(data["correction_original"].mean(0),c+variation,rtol=1e-10,atol=1e-10)
        np.savez_compressed(args.output/f"{arm}_temporal.npz",steps=np.asarray(snapshot_steps),scales=scales,
            query_ids=fixture["query_ids"][:64],mean_function_posterior=b,mean_function_response=r,
            mean_function_correction=c,mean_function_full=r+c,posterior_variation=variation,**data)
        for i,s in enumerate(scales):temporal_rows.append({"arm":arm,"lambda":float(s),
            "mean_snapshot_correction":float(data["correction_original"][:,i].mean()),"mean_function_correction":float(c[i].mean()),
            "posterior_variation":float(variation[i].mean()),"mean_snapshot_full_d_mae":float(np.abs(data["full"][:,i]-1).mean()),
            "mean_of_full_d_mae":float(np.abs(data["full"][:,i].mean(0)-1).mean()),"full_of_mean_d_mae":float(np.abs(r[i]+c[i]-1).mean())})
    for filename,data in (("metrics",rows),("temporal",temporal_rows),("history",history),("fixed_objective_losses",fixed_losses)):
        (args.output/f"{filename}.json").write_text(json.dumps(data,indent=2)+"\n")
        with (args.output/f"{filename}.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    assert all(sha256(p)==h for p,h in hashes.items())
    (args.output/"status.json").write_text(json.dumps({"status":"complete","steps":args.steps,
        "elapsed_seconds":time.monotonic()-start,"all_sources_unchanged":True,
        "actual_snapshots":len(snapshot_steps),"actual_per_step_values":len(value_steps)})+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=Path("artifacts/response_full_20260907/fixed_objective_spaghetti/final.pt"))
    p.add_argument("--cache",type=Path,default=Path("artifacts/response_full_20260907/rb_spaghetti_cache/tape.pt"))
    p.add_argument("--queries",type=Path,default=Path("artifacts/response_full_20260907/ve_spaghetti_shared256/e8_spaghetti_pca/arrays.npz"))
    p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda:0")
    p.add_argument("--steps",type=int,default=256);p.add_argument("--rows",type=int,default=8192)
    p.add_argument("--average-steps",type=int,default=128)
    run(p.parse_args())
