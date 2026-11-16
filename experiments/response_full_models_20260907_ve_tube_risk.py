"""Noisy-query Bayes excess risk for the fixed Exp VE refinement states.

Uses the same test-ID policy, Gaussian seed and three physical scales as the
independent Spaghetti check; there is no model or query selection.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
from experiments.response_full_models_20260907_span import sha256
from experiments.response_full_models_20260907_ve_refinement import load_refinement_field
from experiments.response_full_models_20260907_ve_tube_audit import vector_components
from experiments.response_full_oracle_20260907_geometry import local_frame
from experiments.response_full_oracle_20260907 import tube_posterior
from models.training import load_checkpoint,diffusion_denoiser_output


def run(args):
    torch.set_num_threads(4);args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.run/"manifest.json").read_text())
    source=Path(manifest["source"]);final=args.run/"final.pt"
    hashes={str(p):sha256(p) for p in [source,final,Path(__file__),Path("experiments/response_full_oracle_20260907.py")]}
    initial=load_checkpoint(source,device="cpu");initial.model.double().eval()
    fixture=np.load(args.run/"initial_arrays.npz")
    x=torch.from_numpy(fixture["queries"]).double();rms=initial.normalization_scale
    mean=initial.normalization_mean.double().numpy();normal=local_frame(fixture["query_raw"][:,:3],"tube")
    generator=torch.Generator().manual_seed(2026090703)
    z=torch.randn(x.shape,generator=generator,dtype=torch.float64)
    fields={"initial":initial.model}
    for arm in ["original_lr","low_lr"]:
        for mode in ["last","average"]:fields[arm+"_"+mode]=load_refinement_field(final,arm,mode,"cpu")
    scales=np.array([.01,.0316227766,.316227766]);rows=[];convergence=[]
    arrays={"queries":x.numpy(),"query_ids":fixture["query_ids"],"noise":z.numpy(),"scales":scales}
    for i,s in enumerate(scales):
        observation=x+float(s)*z;raw=observation.numpy()*rms+mean
        truth_raw=np.zeros_like(raw);response=np.zeros(len(x))
        for j,q in enumerate(raw):
            result=tube_posterior(q[:3],float(s*rms))
            truth_raw[j,:3]=result.mean;response[j]=result.response
            if j<4:
                strict=tube_posterior(q[:3],float(s*rms),order=256,cutoff=14.)
                convergence.append({"lambda":float(s),"query_index":j,
                    "response_difference":abs(result.response-strict.response),
                    "posterior_difference_over_sigma":float(np.linalg.norm(result.mean-strict.mean)/(s*rms))})
        truth=(truth_raw-mean)/rms
        arrays[f"oracle_{i}"]=truth;arrays[f"oracle_conditional_variance_{i}"]=response/30
        with torch.no_grad():
            for name,field in fields.items():
                b=diffusion_denoiser_output(field,observation,float(s),preconditioning="edm",data_std=1.).numpy()
                e=b-truth;components=vector_components(e[None],normal,np.array([s]))
                arrays[f"{name}_posterior_{i}"]=b
                row={"state":name,"lambda":float(s),"count":len(x),
                    "risk":float(np.square(b-x.numpy()).mean()/s**2),
                    "continuous_oracle_excess":float(np.square(e).mean()/s**2),
                    "conditional_bayes_variance":float(response.mean()/30),
                    "oracle_realized_risk":float(np.square(truth-x.numpy()).mean()/s**2),
                    **{"excess_"+k:float(v.mean()/30) for k,v in components.items()}}
                rows.append(row);print(json.dumps(row),flush=True)
    assert max(r["response_difference"] for r in convergence)<1e-8
    assert max(r["posterior_difference_over_sigma"] for r in convergence)<1e-8
    np.savez_compressed(args.output/"arrays.npz",**arrays)
    (args.output/"summary.json").write_text(json.dumps(rows,indent=2)+"\n")
    with (args.output/"summary.csv").open("w",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    assert all(sha256(p)==h for p,h in hashes.items())
    (args.output/"manifest.json").write_text(json.dumps({"source_hashes":hashes,
        "query_policy":"Same256 held-out IDs; torchfloat64 standard-normal seed2026090703, same as Spaghetti noisy-risk check",
        "risk_units":"mean squared canonical error divided by30 lambda^2; fixed realization risk is not identically Bayes variance plus excess",
        "component_frame":"Clean source surface tangent/normal; not a claim that noisy-query posterior covariance shares this eigenspace",
        "oracle":"Original uniform-u Exp surface; active3 coordinates integrated at actual noisy observations",
        "convergence":convergence,"all_sources_unchanged":True},indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,default=Path("artifacts/response_full_20260907/ve_exp_refinement"))
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())
