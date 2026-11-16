"""Paired conditional-target-noise removal using a validated population cache.

Both arms share fresh population observations, source weights, fresh AdamW and
the original weighted canonical EDM loss. The sampled clean target versus its
exact posterior mean is the only paired training difference. Equal population
unclipped gradients do not imply equal clipped/Adam updates.
"""
from __future__ import annotations
import argparse,copy,csv,json,time
from pathlib import Path
import numpy as np
import torch
from experiments.response_full_models_20260907_span import sha256
from experiments.response_full_models_20260907_ve_refinement import diagnostics,state_cpu,original_loss
from experiments.response_full_models_20260907_ve import SCALES
from experiments.ve_known_lid_recoverability_campaign import make_splits
from models.training import load_checkpoint,diffusion_denoiser_output


def weighted_target_loss(prediction,target,sigma,data_std=1.):
    per_example=(prediction-target).square().flatten(1).mean(dim=1)
    weights=(sigma.square()+data_std**2)/(sigma.square()*data_std**2)
    return torch.mean(weights*per_example)


def cached_edm_loss(field,observed,sigma,target,data_std=1.):
    prediction=diffusion_denoiser_output(field,observed,sigma,preconditioning="edm",data_std=data_std)
    return weighted_target_loss(prediction,target,sigma,data_std)


def make_pair(source,config):
    fields={arm:copy.deepcopy(source).float() for arm in ["sample_target","posterior_target"]}
    opts={arm:torch.optim.AdamW(field.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
        for arm,field in fields.items()}
    for a,b in zip(fields["sample_target"].parameters(),fields["posterior_target"].parameters()):
        assert torch.equal(a,b) and a.data_ptr()!=b.data_ptr()
    assert all(not opt.state for opt in opts.values())
    return fields,opts


def save_state(path,fields,opts,source,step,averages=None):
    torch.save({"schema":"ve_rao_blackwell_refinement_v1","step":step,"architecture":source["architecture"],
        "training_config":source["training_config"],"normalization":source["normalization"],
        "models":{k:state_cpu(v) for k,v in fields.items()},"optimizers":{k:v.state_dict() for k,v in opts.items()},
        "averages":averages},path)


def run(args):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True,exist_ok=False)
    source=torch.load(args.source,map_location="cpu",weights_only=False)
    cfg=source["training_config"]
    if cfg["diffusion_preconditioning"]!="edm" or cfg["diffusion_loss_weighting"]!="edm_x0" or cfg["diffusion_data_std"]!=1.:
        raise ValueError("Require original preserved EDM contract")
    if cfg["learning_rate"]!=.001 or cfg["dropout"]!=0 or cfg["gradient_clip_norm"]!=1.:
        raise ValueError("Require fixed LR.001/clip1/dropout0 source")
    if args.steps*args.batch_size<=0 or args.average_steps>args.steps or args.average_steps%32:
        raise ValueError("Invalid fixed budget")
    checks_path=args.cache_manifest.with_name("checks.json")
    hashes={str(p):sha256(p) for p in [args.source,args.cache,args.cache_manifest,checks_path,args.queries,Path(__file__),
        Path("models/training.py"),Path("experiments/response_full_models_20260907_ve_refinement.py")]}
    cache_manifest=json.loads(args.cache_manifest.read_text())
    # The owner validates the exact population quadrature before handing off
    # the immutable cache. Completion alone is not an accuracy claim.
    cache_checks=json.loads(checks_path.read_text())
    errors=[v[k] for v in cache_checks for k in ["doubling_error_l2_over_sigma","adaptive_error_l2_over_sigma"]]
    declared_error=cache_manifest["max_independent_posterior_error_l2_over_sigma"]
    if (not errors or not np.isfinite(errors).all() or not np.isfinite(declared_error)
        or max(errors)>2e-6 or declared_error!=max(errors)):
        raise ValueError("Require completed cache with passed independent accuracy checks")
    if cache_manifest["tape_sha256"]!=hashes[str(args.cache)] or cache_manifest["checkpoint_sha256"]!=hashes[str(args.source)]:
        raise ValueError("Immutable cache/source receipt does not match actual files")
    cache=torch.load(args.cache,map_location="cpu",weights_only=False)
    mean=source["normalization"]["mean"].cpu();rms=float(source["normalization"]["scale"])
    torch.testing.assert_close(torch.as_tensor(cache["normalization_mean"]).cpu(),mean,rtol=0,atol=0)
    assert float(cache["normalization_scale"])==rms
    n=args.steps*args.batch_size
    if (cache["dataset"]!="e8_spaghetti_pca" or cache_manifest["rows"]!=n
        or cache["steps"]!=args.steps or cache["batch_size"]!=args.batch_size
        or not cache["fresh_continuous_population"]):
        raise ValueError("The declared fresh no-reuse tape must match the entire fixed run")
    if cache["source_checkpoint_sha256"]!=hashes[str(args.source)]:
        raise ValueError("Source normalization/ancestry does not match cache")
    tape={k:cache[k].to(args.device) for k in ["clean","observed","sigma","oracle_target"]}
    for k,v in tape.items():
        assert v.dtype==torch.float32 and len(v)==n and torch.isfinite(v).all(),k
    assert tape["sigma"].shape==(n,) and tape["clean"].shape==tape["observed"].shape==tape["oracle_target"].shape==(n,30)
    assert float(tape["sigma"].min())>=cfg["sigma_min"] and float(tape["sigma"].max())<=cfg["sigma_max"]
    loaded=load_checkpoint(args.source,device=args.device)
    fields,opts=make_pair(loaded.model,cfg)
    fixture=np.load(args.queries);q=torch.from_numpy(fixture["queries"]).to(args.device)
    _,val_raw,_=make_splits("e8_spaghetti_pca",train_size=100000,validation_size=1000,test_size=1000,data_seed=0)
    validation=((torch.from_numpy(val_raw)-mean)/rms).to(args.device)
    manifest={"source_hashes":hashes,"source_epoch":loaded.best_epoch,"cache_manifest":cache_manifest,
        "steps":args.steps,"batch_size":args.batch_size,"learning_rate":cfg["learning_rate"],"average_steps":args.average_steps,
        "dataset":"Fresh continuous-population Spaghetti tape, not repeated finite100k training set",
        "objective":"Original canonical EDM weight times coordinate-mean squared error; preconditioning/clip unchanged",
        "arms":{"sample_target":"Target sampled cleanX","posterior_target":"Target continuousE[X|Y,sigma]"},
        "sampler":"Contiguous immutable fresh tape batches; no shuffle/reuse; both arms identicalobservedY/sigma",
        "optimizer":"BOTH freshAdamW; same sourceweights, LR.001, clip1, sourceweightdecay",
        "interpretation":"Same populationunclipped mean gradient up to quadrature/roundoff; clipped/Adamupdates neednotmatch; oracletraining is not a deployment proposal",
        "selection":"No LID/checkpoint selection: fixed last, last512weightavg and16function snapshots",
        "snapshot_interval":32,"snapshot_scales":[.01,.1,.316227766,1.],"snapshot_queries":64}
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    save_state(args.output/"step_000000.pt",fields,opts,source,0)
    initial=diagnostics(fields["sample_target"],q,SCALES)
    for key in ["posterior","response","correction_original"]:
        np.testing.assert_allclose(initial[key],fixture[key],rtol=1e-9,atol=1e-9)
    np.savez_compressed(args.output/"initial_arrays.npz",queries=fixture["queries"],query_ids=fixture["query_ids"],scales=SCALES,**initial)
    averages={arm:{k:torch.zeros_like(v,dtype=torch.float64,device="cpu") for k,v in field.state_dict().items()}
        for arm,field in fields.items()}
    snapshots={arm:[] for arm in fields};snapshot_steps=[];history=[];start=time.monotonic()
    for step in range(1,args.steps+1):
        sl=slice((step-1)*args.batch_size,step*args.batch_size)
        observed,sigma,clean,truth=[tape[k][sl] for k in ["observed","sigma","clean","oracle_target"]]
        row={"step":step}
        for arm,field in fields.items():
            field.train();opts[arm].zero_grad(set_to_none=True)
            b=diffusion_denoiser_output(field,observed,sigma,preconditioning="edm",data_std=1.)
            clean_loss=weighted_target_loss(b,clean,sigma);oracle_loss=weighted_target_loss(b,truth,sigma)
            loss=clean_loss if arm=="sample_target" else oracle_loss
            if not torch.isfinite(loss):raise FloatingPointError(arm)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(field.parameters(),cfg["gradient_clip_norm"])
            opts[arm].step()
            row.update({arm+"_objective":float(loss.detach()),arm+"_clean_loss":float(clean_loss.detach()),
                arm+"_oracle_loss":float(oracle_loss.detach()),arm+"_gradient_norm":float(norm),
                arm+"_clip_factor":min(1.,float(cfg["gradient_clip_norm"])/(float(norm)+1e-6))})
            if step>args.steps-args.average_steps:
                for key,v in field.state_dict().items():averages[arm][key].add_(v.detach().cpu().double(),alpha=1/args.average_steps)
        history.append(row)
        if step>args.steps-args.average_steps and (step-(args.steps-args.average_steps))%32==0:
            snapshot_steps.append(step)
            for arm,field in fields.items():snapshots[arm].append(diagnostics(field,q[:64],manifest["snapshot_scales"]))
        if step%128==0:
            print(json.dumps({"elapsed_seconds":time.monotonic()-start,**row}),flush=True)
            (args.output/"status.json").write_text(json.dumps({"status":"training","step":step})+"\n")
        if step in [args.steps//2,args.steps]:save_state(args.output/f"step_{step:06d}.pt",fields,opts,source,step)
    averages={arm:{k:v.float() for k,v in state.items()} for arm,state in averages.items()}
    save_state(args.output/"final.pt",fields,opts,source,args.steps,averages)
    rows=[];temporal_rows=[]
    for arm,field in fields.items():
        for mode in ["last","average"]:
            model=copy.deepcopy(field)
            if mode=="average":model.load_state_dict(averages[arm])
            data=diagnostics(model,q,SCALES);model.eval()
            with torch.no_grad():risk=float(original_loss(model,validation,cfg,torch.Generator(device=args.device).manual_seed(20261209)))
            np.savez_compressed(args.output/f"{arm}_{mode}_arrays.npz",queries=fixture["queries"],query_ids=fixture["query_ids"],scales=SCALES,**data)
            for i,s in enumerate(SCALES):rows.append({"arm":arm,"mode":mode,"lambda":s,
                "response_mean":float(data["response"][i].mean()),"correction_mean":float(data["correction_original"][i].mean()),
                "response_d_mae":float(np.abs(data["response"][i]-1).mean()),"full_d_mae":float(np.abs(data["full"][i]-1).mean()),
                "original_edm_validation_risk":risk})
        sequence=snapshots[arm];data={key:np.stack([item[key] for item in sequence]) for key in sequence[0]}
        b=data["posterior"].mean(0);r=data["response"].mean(0);scales=np.asarray(manifest["snapshot_scales"])
        c=((b-fixture["queries"][:64][None])**2).sum(-1)/scales[:,None]**2
        variation=((data["posterior"]-b[None])**2).sum(-1).mean(0)/scales[:,None]**2
        np.testing.assert_allclose(data["correction_original"].mean(0),c+variation,rtol=1e-10,atol=1e-10)
        np.savez_compressed(args.output/f"{arm}_temporal.npz",steps=np.asarray(snapshot_steps),scales=scales,query_ids=fixture["query_ids"][:64],
            mean_function_posterior=b,mean_function_response=r,mean_function_correction=c,mean_function_full=r+c,posterior_variation=variation,**data)
        for i,s in enumerate(scales):temporal_rows.append({"arm":arm,"lambda":float(s),"mean_snapshot_correction":float(data["correction_original"][:,i].mean()),
            "mean_function_correction":float(c[i].mean()),"posterior_variation":float(variation[i].mean()),
            "mean_snapshot_full_d_mae":float(np.abs(data["full"][:,i]-1).mean()),
            "mean_of_full_d_mae":float(np.abs(data["full"][:,i].mean(0)-1).mean()),
            "full_of_mean_d_mae":float(np.abs(r[i]+c[i]-1).mean())})
    for filename,data in [("metrics",rows),("temporal",temporal_rows),("history",history)]:
        (args.output/f"{filename}.json").write_text(json.dumps(data,indent=2)+"\n")
        with (args.output/f"{filename}.csv").open("w",newline="") as stream:
            w=csv.DictWriter(stream,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    assert all(sha256(p)==h for p,h in hashes.items())
    (args.output/"status.json").write_text(json.dumps({"status":"complete","steps":args.steps,"elapsed_seconds":time.monotonic()-start,
        "all_sources_unchanged":True,"actual_snapshots":len(snapshot_steps)})+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path,required=True);p.add_argument("--cache-manifest",type=Path,required=True)
    p.add_argument("--source",type=Path,default=Path("artifacts/ve_known_lid_recoverability/seed-20260904-prospective/e8_spaghetti_pca/model.pt"))
    p.add_argument("--queries",type=Path,default=Path("artifacts/response_full_20260907/ve_spaghetti_shared256/e8_spaghetti_pca/arrays.npz"))
    p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda:0")
    p.add_argument("--steps",type=int,default=2048);p.add_argument("--batch-size",type=int,default=256);p.add_argument("--average-steps",type=int,default=512)
    run(p.parse_args())
