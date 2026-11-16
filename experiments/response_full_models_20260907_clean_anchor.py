"""One fine-scale clean-anchor objective intervention on stabilized Spaghetti.

Both arms resume identical LAST weights and Adam moments, and retain the full
8192-row noisy posterior-target objective. The extra masked clean-query term
is intentionally different supervision, with unit weight and denominator8192.
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

ARMS=("control","clean_anchor")
ANCHOR_MAX=.03


def clean_anchor_loss(field,clean,sigma):
    """Exactly mean_ALL[mask * original EDM clean reconstruction loss]."""
    mask=sigma<=ANCHOR_MAX
    if not bool(mask.any()):
        return sum(p.sum()*0 for p in field.parameters())
    return cached_edm_loss(field,clean[mask],sigma[mask],clean[mask])*(mask.sum().to(sigma.dtype)/len(sigma))


def resumed_pair(source,optimizer_state):
    fields={arm:copy.deepcopy(source).float() for arm in ARMS};opts={}
    for arm,field in fields.items():
        opt=torch.optim.AdamW(field.parameters(),lr=2e-5)
        opt.load_state_dict(copy.deepcopy(optimizer_state))
        if any(g["lr"]!=2e-5 for g in opt.param_groups):raise ValueError("Parent must already use LR2e-5")
        opts[arm]=opt
    return fields,opts


def save(path,fields,opts,source,step,averages=None):
    torch.save({"schema":"ve_clean_anchor_refinement_v1","step":step,
        "parent_step":source["step"],"architecture":source["architecture"],
        "training_config":source["training_config"],"normalization":source["normalization"],
        "models":{k:state_cpu(v) for k,v in fields.items()},
        "optimizers":{k:v.state_dict() for k,v in opts.items()},"averages":averages},path)


def write_rows(output,name,rows):
    (output/f"{name}.json").write_text(json.dumps(rows,indent=2)+"\n")
    with (output/f"{name}.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(args):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True,exist_ok=False)
    source=torch.load(args.source,map_location="cpu",weights_only=False);cfg=source["training_config"]
    if source["schema"]!="ve_fixed_objective_lr_refinement_v1" or source["step"]!=256:
        raise ValueError("Require completed deterministic LR pair LAST source")
    if (args.rows!=8192 or args.steps<1 or args.average_steps>args.steps or args.average_steps%16
        or cfg["diffusion_preconditioning"]!="edm" or cfg["diffusion_data_std"]!=1.
        or cfg["dropout"]!=0 or cfg["gradient_clip_norm"]!=1.):raise ValueError("Invalid declared protocol")
    source_manifest=args.source.with_name("manifest.json");source_status=args.source.with_name("status.json")
    row_ids_path=args.source.with_name("objective_row_ids.npy")
    if json.loads(source_status.read_text())["status"]!="complete":raise ValueError("Incomplete parent")
    old_manifest=json.loads(source_manifest.read_text())
    inputs=[args.source,args.cache,args.queries,source_manifest,source_status,row_ids_path,Path(__file__),
        Path("experiments/response_full_models_20260907_rb_training.py"),
        Path("experiments/response_full_models_20260907_ve_refinement.py"),
        Path("experiments/response_full_models_20260907_ve.py"),
        Path("experiments/fm_mechanism_fast_trace.py"),
        Path("experiments/ve_known_lid_recoverability_campaign.py"),Path("models/training.py"),Path("models/neural_fields.py")]
    hashes={str(p):sha256(p) for p in inputs}
    if old_manifest["source_hashes"][str(args.cache)]!=hashes[str(args.cache)]:raise ValueError("Changed cache")
    cache=torch.load(args.cache,map_location="cpu",weights_only=False)
    ids=torch.arange(len(cache["sigma"])-args.rows,len(cache["sigma"]))
    np.testing.assert_array_equal(ids.numpy(),np.load(row_ids_path))
    tape={k:cache[k][ids].to(args.device) for k in ("clean","observed","sigma","oracle_target")}
    mean=source["normalization"]["mean"].cpu();rms=float(source["normalization"]["scale"])
    torch.testing.assert_close(torch.as_tensor(cache["normalization_mean"]),mean,rtol=0,atol=0)
    assert float(cache["normalization_scale"])==rms
    assert cache["dataset"]=="e8_spaghetti_pca" and cache["fresh_continuous_population"]
    field=load_refinement_field(args.source,"low_lr","last",args.device).float()
    fields,opts=resumed_pair(field,source["optimizers"]["low_lr"])
    for opt in opts.values():
        assert all(float(v["step"])==2816 for v in opt.state.values())
        assert all(g["weight_decay"]==cfg["weight_decay"] for g in opt.param_groups)
    fixture=np.load(args.queries);q=torch.from_numpy(fixture["queries"]).to(args.device)
    _,val_raw,_=make_splits("e8_spaghetti_pca",train_size=100000,validation_size=1000,test_size=1000,data_seed=0)
    validation=((torch.from_numpy(val_raw)-mean)/rms).to(args.device)
    rng_before=torch.get_rng_state().clone()
    cuda_rng_before=torch.cuda.get_rng_state(args.device).clone() if str(args.device).startswith("cuda") else None
    selected=int((tape["sigma"]<=ANCHOR_MAX).sum())
    manifest={"source_hashes":hashes,"parent_arm":"low_lr","parent_mode":"last",
        "optimizer":"Both deep-copy parent LAST weights and Adam moments/counter2816; LR2e-5, clip1 and weight decay unchanged",
        "rows":args.rows,"row_start_in_parent_tape":int(ids[0]),"row_stop_exclusive":len(cache["sigma"]),
        "base_objective":"Original EDM-weighted posterior-target mean over same fixed8192 noisy observations",
        "anchor_objective":"mean_ALL8192[1{sigma<=.03} * w_EDM(sigma) * ||b(cleanX,sigma)-cleanX||^2/D]",
        "anchor_coefficient":1.,"anchor_max_sigma":ANCHOR_MAX,"anchor_selected_rows":selected,
        "anchor_fraction":selected/args.rows,"steps":args.steps,"learning_rate":2e-5,"full_batch_size":args.rows,
        "sampling":"No resampling in either arm; additional clean coordinates reuse same cached X and sigma",
        "average_steps":args.average_steps,"snapshot_interval":16,"snapshot_scales":[.01,.1,.316227766,1.],"snapshot_queries":64,
        "scope":"Intentional finite-scale objective change on resolved Spaghetti; not Bayes-equivalent supervision or boundary transfer",
        "selection":"Fixed budget, last and final256 weight average; no LID/model/scale selection",
        "source_training_config":"Inherited ancestry only; actual execution specified by this manifest"}
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    np.save(args.output/"objective_row_ids.npy",ids.numpy());save(args.output/"step_000000.pt",fields,opts,source,0)
    initial=diagnostics(fields["control"],q,SCALES);old=np.load(args.source.with_name("low_lr_last_arrays.npz"))
    for k in ("posterior","response","correction_original"):np.testing.assert_allclose(initial[k],old[k],rtol=1e-9,atol=1e-9)
    np.savez_compressed(args.output/"initial_arrays.npz",queries=fixture["queries"],query_ids=fixture["query_ids"],scales=SCALES,**initial)
    averages={a:{k:torch.zeros_like(v,dtype=torch.float64,device="cpu") for k,v in f.state_dict().items()} for a,f in fields.items()}
    snapshots={a:[] for a in ARMS};snapshot_steps=[];history=[];fixed_losses=[];start=time.monotonic()
    def fixed_loss_row(step):
        row={"step":step}
        with torch.no_grad():
            for arm,model in fields.items():
                model.eval();row[arm+"_base"]=float(cached_edm_loss(model,tape["observed"],tape["sigma"],tape["oracle_target"]))
                row[arm+"_anchor"]=float(clean_anchor_loss(model,tape["clean"],tape["sigma"]))
        fixed_losses.append(row)
    fixed_loss_row(0)
    for step in range(1,args.steps+1):
        row={"step":step}
        for arm,model in fields.items():
            model.train();opts[arm].zero_grad(set_to_none=True)
            base=cached_edm_loss(model,tape["observed"],tape["sigma"],tape["oracle_target"])
            anchor=clean_anchor_loss(model,tape["clean"],tape["sigma"]) if arm=="clean_anchor" else base.new_zeros(())
            loss=base+anchor
            if not torch.isfinite(loss):raise FloatingPointError(arm)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["gradient_clip_norm"]);opts[arm].step()
            row.update({arm+"_base":float(base.detach()),arm+"_anchor_added":float(anchor.detach()),
                arm+"_total":float(loss.detach()),arm+"_gradient_norm":float(norm),
                arm+"_clip_factor":min(1.,float(cfg["gradient_clip_norm"])/(float(norm)+1e-6))})
            if step>args.steps-args.average_steps:
                for k,v in model.state_dict().items():averages[arm][k].add_(v.detach().cpu().double(),alpha=1/args.average_steps)
        history.append(row)
        if step>args.steps-args.average_steps and (step-(args.steps-args.average_steps))%16==0:
            snapshot_steps.append(step)
            for arm,model in fields.items():snapshots[arm].append(diagnostics(model,q[:64],manifest["snapshot_scales"]))
        if step%128==0:
            fixed_loss_row(step);print(json.dumps({"elapsed_seconds":time.monotonic()-start,**row,"fixed_losses":fixed_losses[-1]}),flush=True)
            (args.output/"status.json").write_text(json.dumps({"status":"training","step":step})+"\n")
            save(args.output/f"step_{step:06d}.pt",fields,opts,source,step)
    averages={a:{k:v.float() for k,v in state.items()} for a,state in averages.items()}
    save(args.output/"final.pt",fields,opts,source,args.steps,averages)
    assert torch.equal(torch.get_rng_state(),rng_before)
    if cuda_rng_before is not None:assert torch.equal(torch.cuda.get_rng_state(args.device),cuda_rng_before)
    torch.save({"cpu":rng_before,"cuda":cuda_rng_before,"unchanged":True},args.output/"rng_receipt.pt")
    rows=[];temporal_rows=[]
    for arm,model in fields.items():
        for mode in ("last","average"):
            evaluated=copy.deepcopy(model)
            if mode=="average":evaluated.load_state_dict(averages[arm])
            data=diagnostics(evaluated,q,SCALES);evaluated.eval()
            with torch.no_grad():
                risk=float(original_loss(evaluated,validation,cfg,torch.Generator(device=args.device).manual_seed(20261209)))
                objective=float(cached_edm_loss(evaluated,tape["observed"],tape["sigma"],tape["oracle_target"]))
                anchor=float(clean_anchor_loss(evaluated,tape["clean"],tape["sigma"]))
            np.savez_compressed(args.output/f"{arm}_{mode}_arrays.npz",queries=fixture["queries"],query_ids=fixture["query_ids"],scales=SCALES,**data)
            for i,s in enumerate(SCALES):rows.append({"arm":arm,"mode":mode,"lambda":s,
                "response_mean":float(data["response"][i].mean()),"correction_mean":float(data["correction_original"][i].mean()),
                "response_d_mae":float(np.abs(data["response"][i]-1).mean()),"full_d_mae":float(np.abs(data["full"][i]-1).mean()),
                "original_edm_validation_risk":risk,"fixed_base_loss":objective,"fixed_anchor_loss":anchor})
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
    for name,data in (("metrics",rows),("temporal",temporal_rows),("history",history),("fixed_objective_losses",fixed_losses)):write_rows(args.output,name,data)
    assert all(sha256(p)==h for p,h in hashes.items())
    (args.output/"status.json").write_text(json.dumps({"status":"complete","steps":args.steps,
        "elapsed_seconds":time.monotonic()-start,"all_sources_unchanged":True,"actual_snapshots":len(snapshot_steps)})+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=Path("artifacts/response_full_20260907/fixed_objective_lr_spaghetti/final.pt"))
    p.add_argument("--cache",type=Path,default=Path("artifacts/response_full_20260907/rb_spaghetti_cache/tape.pt"))
    p.add_argument("--queries",type=Path,default=Path("artifacts/response_full_20260907/ve_spaghetti_shared256/e8_spaghetti_pca/arrays.npz"))
    p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda:0")
    p.add_argument("--steps",type=int,default=1024);p.add_argument("--rows",type=int,default=8192)
    p.add_argument("--average-steps",type=int,default=256);run(p.parse_args())
