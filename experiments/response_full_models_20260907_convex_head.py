"""One SVD final-head refit on the unchanged frozen posterior-target objective.

All nonlinear features are frozen. No ridge, model/scale selection, or optimizer
updates occur. EDM data_std1 cancels weighting exactly in residual-head units.
"""
from __future__ import annotations
import argparse,copy,csv,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from experiments.response_full_models_20260907_span import sha256
from experiments.response_full_models_20260907_rb_training import cached_edm_loss,weighted_target_loss
from experiments.response_full_models_20260907_ve_refinement import diagnostics,load_refinement_field,state_cpu
from experiments.response_full_models_20260907_ve import SCALES
from experiments.response_full_models_20260907_spaghetti_local import active_jacobian,tangent_from_phase,split_jacobian_error
from experiments.response_full_models_20260907_ve_refinement_audit import value_components
from experiments.response_full_models_20260907_oracle_audit import decompose
from experiments.ve_known_lid_recoverability_campaign import make_splits
from models.training import diffusion_denoiser_output,_sample_log_uniform

HEAD_KEYS={"output.2.weight","output.2.bias"}


def body_hash(field):
    digest=hashlib.sha256()
    for name,value in sorted(field.state_dict().items()):
        if name in HEAD_KEYS:continue
        digest.update(name.encode());digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape,dtype=np.int64).tobytes())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def capture_features(field,observed,sigma,batch_size=1024):
    captured=[]
    handle=field.output[-1].register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
    try:
        for start in range(0,len(observed),batch_size):
            diffusion_denoiser_output(field,observed[start:start+batch_size],sigma[start:start+batch_size],
                preconditioning="edm",data_std=1.)
    finally:handle.remove()
    return torch.cat(captured)


def residual_targets(observed,sigma,posterior_target):
    denom=1+sigma.square();skip=1/denom;cout=sigma/denom.sqrt()
    return (posterior_target-skip[:,None]*observed)/cout[:,None]


def solve_head(features,targets):
    """Float64 LAPACK SVD, rank cutoff fixed by machine epsilon and matrix size."""
    assert features.dtype==targets.dtype==torch.float64
    design=torch.cat([features.cpu(),torch.ones(len(features),1,dtype=torch.float64)],dim=1)
    target=targets.cpu();rcond=torch.finfo(torch.float64).eps*max(design.shape)
    result=torch.linalg.lstsq(design,target,rcond=rcond,driver="gelsd")
    coeff=result.solution;residual=design@coeff-target;gradient=design.T@residual
    spectrum=result.singular_values;rank=int(result.rank)
    assert rank==int((spectrum>rcond*spectrum[0]).sum())
    stationarity=float(gradient.norm()/(design.norm()*residual.norm()).clamp_min(1e-300))
    receipt={"solver":"torch.linalg.lstsq CPU float64 LAPACK gelsd (SVD)","rcond":float(rcond),
        "cutoff_rule":"float64 machine epsilon times max(rows,columns), fixed before fit",
        "singular_cutoff_absolute":float(rcond*spectrum[0]),"rank":rank,
        "rows":design.shape[0],"columns":design.shape[1],"outputs":target.shape[1],
        "retained_condition_number":float(spectrum[0]/spectrum[rank-1]),
        "full_condition_number":float(spectrum[0]/spectrum[-1]) if spectrum[-1]>0 else None,
        "coefficient_frobenius_norm":float(coeff.norm()),"normalized_stationarity":stationarity,
        "raw_stationarity_max_abs":float(gradient.abs().max()),
        "residual_mse":float(residual.square().mean()),"residual_mse_per_output":residual.square().mean(0).tolist()}
    return coeff,design,spectrum,residual,receipt


def replaced_head(field,coefficient,round_head=False):
    result=copy.deepcopy(field)
    values=coefficient.float().double() if round_head else coefficient
    with torch.no_grad():
        result.output[-1].weight.copy_(values[:-1].T.to(result.output[-1].weight))
        result.output[-1].bias.copy_(values[-1].to(result.output[-1].bias))
    assert body_hash(result)==body_hash(field)
    return result


@torch.no_grad()
def noisy_validation_tape(source,device):
    _,validation,_=make_splits("e8_spaghetti_pca",train_size=100000,validation_size=1000,test_size=1000,data_seed=0)
    mean=source["normalization"]["mean"].float();rms=float(source["normalization"]["scale"])
    x=((torch.from_numpy(validation)-mean)/rms).to(device)
    rng=torch.Generator(device=device).manual_seed(20261209);cfg=source["training_config"]
    sigma=_sample_log_uniform(len(x),minimum=cfg["sigma_min"],maximum=cfg["sigma_max"],data=x,generator=rng)
    z=torch.randn(x.shape,dtype=x.dtype,device=device,generator=rng)
    return {"clean":x.double(),"observed":(x+sigma[:,None]*z).double(),"sigma":sigma.double()}


def run(args):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    parent=torch.load(args.source,map_location="cpu",weights_only=False)
    if parent["schema"]!="ve_fixed_objective_lr_refinement_v1":raise ValueError("Wrong source stage")
    cfg=parent["training_config"]
    if cfg["diffusion_preconditioning"]!="edm" or cfg["diffusion_data_std"]!=1. or cfg["dropout"]!=0:
        raise ValueError("Require preserved EDM data_std1, zero-dropout source")
    parent_manifest=json.loads(args.source.with_name("manifest.json").read_text())
    row_path=args.source.with_name("objective_row_ids.npy");ids=np.load(row_path)
    if len(ids)!=8192 or not np.array_equal(ids,np.arange(516096,524288)):raise ValueError("Require original fixed8192 objective rows")
    inputs=[args.source,args.cache,args.queries,args.oracle,args.noisy_oracle,row_path,args.source.with_name("manifest.json"),Path(__file__),
        Path("models/training.py"),Path("models/neural_fields.py"),Path("experiments/response_full_models_20260907_ve_refinement.py"),
        Path("experiments/response_full_models_20260907_rb_training.py"),Path("experiments/response_full_models_20260907_spaghetti_local.py"),
        Path("experiments/response_full_models_20260907_oracle_audit.py"),Path("experiments/response_full_models_20260907_ve_refinement_audit.py"),
        Path("experiments/fm_mechanism_fast_trace.py"),Path("experiments/ve_known_lid_recoverability_campaign.py"),
        Path("experiments/response_full_models_20260907_ve.py"),Path("experiments/response_full_models_20260907_span.py")]
    hashes={str(p):sha256(p) for p in inputs}
    assert parent_manifest["source_hashes"][str(args.cache)]==hashes[str(args.cache)]
    cache=torch.load(args.cache,map_location="cpu",weights_only=False)
    tape={k:cache[k][ids].to(args.device).double() for k in ("observed","sigma","oracle_target")}
    torch.testing.assert_close(torch.as_tensor(cache["normalization_mean"]),parent["normalization"]["mean"],rtol=0,atol=0)
    assert float(cache["normalization_scale"])==float(parent["normalization"]["scale"])
    field=load_refinement_field(args.source,"low_lr","average",args.device).requires_grad_(False)
    before_body=body_hash(field);features=capture_features(field,tape["observed"],tape["sigma"])
    target=residual_targets(tape["observed"],tape["sigma"],tape["oracle_target"])
    coefficient,design,spectrum,residual,solve=solve_head(features,target)
    old_coefficient=torch.cat([field.output[-1].weight.T,field.output[-1].bias[None]],dim=0).cpu()
    old_residual=design@old_coefficient-target.cpu()
    with torch.no_grad():old_objective=float(cached_edm_loss(field,tape["observed"],tape["sigma"],tape["oracle_target"]))
    np.testing.assert_allclose(old_objective,float(old_residual.square().mean()),rtol=1e-10,atol=1e-12)
    if solve["residual_mse"]>old_objective*(1+1e-8):raise ArithmeticError("Convex fit worsens fixed data objective")
    models={"original":field,"refit":replaced_head(field,coefficient),"refit_head_float32":replaced_head(field,coefficient,True)}
    solve.update({"old_coefficient_frobenius_norm":float(old_coefficient.norm()),
        "new_to_old_coefficient_norm_ratio":float(coefficient.norm()/old_coefficient.norm()),
        "old_weight_norm":float(old_coefficient[:-1].norm()),"new_weight_norm":float(coefficient[:-1].norm()),
        "old_bias_norm":float(old_coefficient[-1].norm()),"new_bias_norm":float(coefficient[-1].norm()),
        "old_fixed_data_objective":old_objective})
    torch.save({"schema":"ve_convex_head_refit_v1","architecture":parent["architecture"],
        "training_config":parent["training_config"],"normalization":parent["normalization"],
        "models":{k:state_cpu(v) for k,v in models.items()},"averages":None,
        "source_mode":"low_lr fixed parameter average","method":"One unregularized SVD affine-head solve; no optimizer/training steps"},args.output/"models.pt")
    np.savez_compressed(args.output/"least_squares.npz",design=design.numpy(),targets=target.cpu().numpy(),
        coefficient=coefficient.numpy(),old_coefficient=old_coefficient.numpy(),singular_values=spectrum.numpy(),
        residual=residual.numpy(),objective_row_ids=ids)
    fixture=np.load(args.queries);oracle=np.load(args.oracle);q=torch.from_numpy(fixture["queries"]).double().to(args.device)
    for key in ("queries","query_ids"):np.testing.assert_array_equal(fixture[key],oracle[key])
    np.testing.assert_allclose(SCALES,oracle["scales"],rtol=1e-9,atol=0)
    tangent=tangent_from_phase(oracle["query_phase"]);validation=noisy_validation_tape(parent,args.device)
    noisy=np.load(args.noisy_oracle);noisy_q=torch.from_numpy(noisy["queries"]).double().to(args.device)
    noisy_z=torch.from_numpy(noisy["noise"]).double().to(args.device)
    rows=[];risk_rows=[];objective_rows=[]
    for name,model in models.items():
        assert body_hash(model)==before_body
        data=diagnostics(model,q,SCALES)
        if name=="original":
            old=np.load(args.source.with_name("low_lr_average_arrays.npz"))
            for key in ("posterior","response","correction_original"):np.testing.assert_allclose(data[key],old[key],rtol=1e-9,atol=1e-9)
        with torch.no_grad():
            vpred=diffusion_denoiser_output(model,validation["observed"],validation["sigma"],preconditioning="edm",data_std=1.)
            risk=float(weighted_target_loss(vpred,validation["clean"],validation["sigma"]))
            objective=float(cached_edm_loss(model,tape["observed"],tape["sigma"],tape["oracle_target"]))
        if name=="refit":np.testing.assert_allclose(objective,solve["residual_mse"],rtol=1e-8,atol=1e-12)
        objective_rows.append({"state":name,"fixed_data_objective":objective,"original_noisy_validation_risk":risk})
        result=decompose(fixture["queries"],data["posterior"],oracle["posterior"],data["response"],oracle["response"],SCALES)
        parts=value_components(result["posterior_error"],tangent,SCALES);jarrays={}
        for i,s in enumerate(SCALES):
            value,j=active_jacobian(lambda x:diffusion_denoiser_output(model,x,float(s),preconditioning="edm",data_std=1.),q,active=30)
            value,j=value.cpu().numpy(),j.cpu().numpy()
            np.testing.assert_allclose(value,data["posterior"][i],rtol=1e-9,atol=1e-9)
            np.testing.assert_allclose(np.trace(j,axis1=-2,axis2=-1),data["response"][i],rtol=1e-9,atol=1e-9)
            active=j[:,:20,:20];truth=oracle["jacobian_active"][i]
            tangent_r=np.einsum("bi,bij,bj->b",tangent,active,tangent)
            extra={"jacobian":j,"response_tangent":tangent_r,
                "response_curve_normal":np.trace(active,axis1=-2,axis2=-1)-tangent_r,
                "response_ambient_normal":np.trace(j[:,20:,20:],axis1=-2,axis2=-1),
                **split_jacobian_error(active-truth,tangent)}
            for k,v in extra.items():jarrays.setdefault(k,[]).append(v)
            rows.append({"state":name,"lambda":s,"count":len(q),"response_mean":float(data["response"][i].mean()),
                "correction_mean":float(data["correction_original"][i].mean()),"full_d_mae":float(abs(data["full"][i]-1).mean()),
                "full_oracle_mae":float(abs(result["full"][i]-result["oracle_full"][i]).mean()),
                "response_oracle_mae":float(abs(result["response_error"][i]).mean()),
                "posterior_error_energy":float(result["posterior_error_energy"][i].mean()),
                **{k+"_mean":float(v.mean()) for k,v in extra.items() if v.ndim==1},
                **{"posterior_error_"+k+"_energy":float(v[i].mean()) for k,v in parts.items()}})
        np.savez_compressed(args.output/f"{name}_arrays.npz",queries=fixture["queries"],query_ids=fixture["query_ids"],scales=SCALES,
            **data,**{k:np.stack(v) for k,v in jarrays.items()},**{"posterior_error_"+k+"_energy":v for k,v in parts.items()},
            posterior_error_energy=result["posterior_error_energy"],oracle_cross=result["oracle_cross"])
        noisy_predictions=[]
        for i,s in enumerate(noisy["scales"]):
            with torch.no_grad():b=diffusion_denoiser_output(model,noisy_q+float(s)*noisy_z,float(s),preconditioning="edm",data_std=1.)
            reference=torch.from_numpy(noisy[f"oracle_{i}"]).to(args.device)
            noisy_predictions.append(b.cpu().numpy())
            risk_rows.append({"state":name,"lambda":float(s),"count":len(noisy_q),
                "noisy_clean_mse_over_Dlambda2":float((b-noisy_q).square().mean()/s**2),
                "noisy_oracle_error_over_Dlambda2":float((b-reference).square().mean()/s**2)})
        np.savez_compressed(args.output/f"{name}_noisy.npz",posterior=np.stack(noisy_predictions),queries=noisy["queries"],noise=noisy["noise"],scales=noisy["scales"])
    for name,items in (("metrics",rows),("noisy_risk",risk_rows),("objectives",objective_rows)):
        (args.output/f"{name}.json").write_text(json.dumps(items,indent=2)+"\n")
        with (args.output/f"{name}.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
    (args.output/"solve.json").write_text(json.dumps(solve,indent=2)+"\n")
    assert all(sha256(p)==h for p,h in hashes.items())
    (args.output/"manifest.json").write_text(json.dumps({"sources":hashes,"source_body_sha256":before_body,
        "body_sha256_by_state":{k:body_hash(v) for k,v in models.items()},"target":"Same fixed8192 continuous posterior targets, cachedfloat32 promoted tofloat64",
        "objective_identity":"EDM data_std1: w*cout²=1, affine-head feature MSE equals original denoising DATA loss",
        "regularization_scope":"Unregularized convex DATA objective, not AdamW weight-decay regularization or optimizer trajectory",
        "precision":"Float64 original/refit inference; sensitivity state rounds ONLY fitted head coefficients tofloat32 then evaluatesfloat64, not fullfloat32 execution",
        "selection":"One machine-precision rank cutoff; no ridge/LID/scale/checkpoint selection",
        "noisy_risk_scope":"Same fixed independent continuous-oracle noisy queries; empirical squared model-oracle error estimates population excess, not an exact finite-sample Bayes identity",
        "completed":True,"all_sources_unchanged":True,"elapsed_seconds":time.monotonic()-start},indent=2)+"\n")
    print(json.dumps({"solve":solve,"objectives":objective_rows,"noisy_risk":risk_rows}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=Path("artifacts/response_full_20260907/fixed_objective_lr_spaghetti/final.pt"))
    p.add_argument("--cache",type=Path,default=Path("artifacts/response_full_20260907/rb_spaghetti_cache/tape.pt"))
    p.add_argument("--queries",type=Path,default=Path("artifacts/response_full_20260907/ve_spaghetti_shared256/e8_spaghetti_pca/arrays.npz"))
    p.add_argument("--oracle",type=Path,default=Path("artifacts/response_full_20260907/population_oracle_ve_spaghetti256/arrays.npz"))
    p.add_argument("--noisy-oracle",type=Path,default=Path("../notes/response_full_20260907/spaghetti_noisy_risk/arrays.npz"))
    p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda:0")
    run(p.parse_args())
