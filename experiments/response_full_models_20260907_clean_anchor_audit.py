"""Held-out value, exact-Jacobian and noisy-risk audit of the clean-anchor pair."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from experiments.response_full_models_20260907_span import sha256
from experiments.response_full_models_20260907_ve_refinement import load_refinement_field
from experiments.response_full_models_20260907_spaghetti_local import active_jacobian,tangent_from_phase,split_jacobian_error
from experiments.response_full_models_20260907_ve_refinement_audit import value_components
from experiments.response_full_models_20260907_oracle_audit import decompose
from experiments.response_full_models_20260907_clean_anchor import write_rows
from models.training import diffusion_denoiser_output


def run(args):
    torch.set_num_threads(4);args.output.mkdir(parents=True,exist_ok=False)
    if json.loads((args.run/"status.json").read_text())["status"]!="complete":raise ValueError("Incomplete paired run")
    inputs=[args.run/"final.pt",args.run/"step_000000.pt",args.run/"manifest.json",args.run/"status.json",args.oracle,args.noisy_oracle,Path(__file__),
        Path("experiments/response_full_models_20260907_span.py"),
        Path("experiments/response_full_models_20260907_ve_refinement.py"),
        Path("experiments/response_full_models_20260907_spaghetti_local.py"),
        Path("experiments/response_full_models_20260907_ve_refinement_audit.py"),
        Path("experiments/response_full_models_20260907_oracle_audit.py"),
        Path("experiments/response_full_models_20260907_clean_anchor.py"),Path("models/training.py"),Path("models/neural_fields.py")]
    hashes={str(p):sha256(p) for p in inputs};oracle=np.load(args.oracle);noisy=np.load(args.noisy_oracle)
    tangent=tangent_from_phase(oracle["query_phase"]);rows=[];risks=[];temporal_rows=[]
    noisy_q=torch.from_numpy(noisy["queries"]).double().to(args.device);noisy_z=torch.from_numpy(noisy["noise"]).double().to(args.device)
    cases=[("initial","control","last",args.run/"step_000000.pt",args.run/"initial_arrays.npz")]
    cases.extend((f"{a}_{m}",a,m,args.run/"final.pt",args.run/f"{a}_{m}_arrays.npz") for a in ("control","clean_anchor") for m in ("last","average"))
    for name,arm,mode,checkpoint,arrays_path in cases:
        hashes[str(arrays_path)]=sha256(arrays_path);saved=np.load(arrays_path);scales=saved["scales"]
        for key in ("queries","query_ids"):np.testing.assert_array_equal(saved[key],oracle[key])
        np.testing.assert_allclose(scales,oracle["scales"],rtol=1e-9,atol=0)
        q=torch.from_numpy(saved["queries"]).double().to(args.device)
        field=load_refinement_field(checkpoint,arm,mode,args.device)
        result=decompose(saved["queries"],saved["posterior"],oracle["posterior"],saved["response"],oracle["response"],scales)
        parts=value_components(result["posterior_error"],tangent,scales);accumulated={}
        for i,s in enumerate(scales):
            value,j=active_jacobian(lambda x:diffusion_denoiser_output(field,x,float(s),preconditioning="edm",data_std=1.),q,active=30)
            value,j=value.cpu().numpy(),j.cpu().numpy()
            np.testing.assert_allclose(value,saved["posterior"][i],rtol=1e-9,atol=1e-9)
            np.testing.assert_allclose(np.trace(j,axis1=-2,axis2=-1),saved["response"][i],rtol=1e-9,atol=1e-9)
            active=j[:,:20,:20];truth=oracle["jacobian_active"][i]
            tr=np.einsum("bi,bij,bj->b",tangent,active,tangent);trtrue=np.einsum("bi,bij,bj->b",tangent,truth,tangent)
            extra={"jacobian":j,"response_tangent":tr,"response_tangent_error":tr-trtrue,
                "response_curve_normal":np.trace(active,axis1=-2,axis2=-1)-tr,
                "response_curve_normal_error":np.trace(active-truth,axis1=-2,axis2=-1)-tr+trtrue,
                "response_ambient_normal":np.trace(j[:,20:,20:],axis1=-2,axis2=-1),
                **split_jacobian_error(active-truth,tangent)}
            for k,v in extra.items():accumulated.setdefault(k,[]).append(v)
            rows.append({"state":name,"lambda":float(s),"count":len(q),
                "response_mean":float(saved["response"][i].mean()),"correction_mean":float(saved["correction_original"][i].mean()),
                "response_d_mae":float(abs(saved["response"][i]-1).mean()),"full_d_mae":float(abs(saved["full"][i]-1).mean()),
                "full_oracle_mae":float(abs(result["full"][i]-result["oracle_full"][i]).mean()),
                "response_oracle_mae":float(abs(result["response_error"][i]).mean()),
                "signed_response_error":float(result["response_error"][i].mean()),
                **{k+"_mean":float(v[i].mean()) for k,v in result.items() if v.ndim==2},
                **{k+"_mean":float(v.mean()) for k,v in extra.items() if v.ndim==1},
                **{"posterior_error_"+k+"_energy":float(v[i].mean()) for k,v in parts.items()}})
        np.savez_compressed(args.output/f"{name}.npz",scales=scales,query_ids=saved["query_ids"],queries=saved["queries"],
            **result,**{k:np.stack(v) for k,v in accumulated.items()},**{"posterior_error_"+k+"_energy":v for k,v in parts.items()})
        preds=[]
        for i,s in enumerate(noisy["scales"]):
            with torch.no_grad():b=diffusion_denoiser_output(field,noisy_q+float(s)*noisy_z,float(s),preconditioning="edm",data_std=1.)
            truth=torch.from_numpy(noisy[f"oracle_{i}"]).to(args.device);preds.append(b.cpu().numpy())
            risks.append({"state":name,"lambda":float(s),"count":len(noisy_q),
                "noisy_clean_mse_over_Dlambda2":float((b-noisy_q).square().mean()/s**2),
                "noisy_oracle_error_over_Dlambda2":float((b-truth).square().mean()/s**2)})
        np.savez_compressed(args.output/f"{name}_noisy.npz",posterior=np.stack(preds),queries=noisy["queries"],noise=noisy["noise"],scales=noisy["scales"])
    for arm in ("control","clean_anchor"):
        path=args.run/f"{arm}_temporal.npz";hashes[str(path)]=sha256(path);seq=np.load(path)
        variation=seq["posterior"]-seq["mean_function_posterior"][None]
        parts=value_components(variation,tangent[:64],seq["scales"])
        np.testing.assert_allclose(sum(parts.values()).mean(0),seq["posterior_variation"],rtol=1e-10,atol=1e-10)
        for i,s in enumerate(seq["scales"]):temporal_rows.append({"arm":arm,"lambda":float(s),
            "mean_function_correction":float(seq["mean_function_correction"][i].mean()),
            "posterior_variation":float(seq["posterior_variation"][i].mean()),
            **{"variation_"+k:float(v[:,i].mean()) for k,v in parts.items()}})
    for name,data in (("summary",rows),("noisy_risk",risks),("temporal",temporal_rows)):write_rows(args.output,name,data)
    assert all(sha256(p)==h for p,h in hashes.items())
    (args.output/"manifest.json").write_text(json.dumps({"sources":hashes,"completed":True,
        "scope":"Original noisy loss retained; treatment changes objective with fine-scale clean reconstruction. Exact J is independent ordinary autograd replay.",
        "geometry":"True1D tangent,19 within-span normals,10 ambient null directions; labels not used in training",
        "noisy_risk":"Same fixed independent population-oracle tape: squared model-oracle error estimates excess risk, not exact finite-bank Bayes difference",
        "all_sources_unchanged":True},indent=2)+"\n")
    print(json.dumps({"noisy_risk":risks,"temporal":temporal_rows}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",type=Path,required=True)
    p.add_argument("--oracle",type=Path,default=Path("artifacts/response_full_20260907/population_oracle_ve_spaghetti256/arrays.npz"))
    p.add_argument("--noisy-oracle",type=Path,default=Path("../notes/response_full_20260907/spaghetti_noisy_risk/arrays.npz"))
    p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda:0");run(p.parse_args())
