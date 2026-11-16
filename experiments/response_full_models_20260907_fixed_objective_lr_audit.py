"""Exact population, local-Jacobian and temporal audit of the Rao--Blackwell target control.

This is a fixed-run diagnostic, not another training intervention or selector.
Local tangent directions and population posteriors are oracle diagnostics.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch

from experiments.response_full_models_20260907_span import sha256
from experiments.response_full_models_20260907_spaghetti_local import active_jacobian,split_jacobian_error,tangent_from_phase
from experiments.response_full_models_20260907_oracle_audit import decompose
from experiments.response_full_models_20260907_ve_refinement import load_refinement_field
from models.training import diffusion_denoiser_output


def value_components(error,tangent,scales):
    """Supports [scale,query,D] or [snapshot,scale,query,D] errors."""
    s2=np.asarray(scales)[:,None]**2
    tangent_energy=np.einsum("...qd,qd->...q",error[...,:20],tangent)**2/s2
    span_energy=(error[...,:20]**2).sum(-1)/s2
    ambient=(error[...,20:]**2).sum(-1)/s2
    result={"tangent":tangent_energy,"curve_normal":span_energy-tangent_energy,"ambient_normal":ambient}
    np.testing.assert_allclose(sum(result.values()),(error**2).sum(-1)/s2,rtol=1e-11,atol=1e-11)
    return result


def run(args):
    torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=False)
    checkpoint=args.run/"final.pt";before=sha256(checkpoint)
    oracle=np.load(args.oracle);rows=[];temporal_rows=[];sources={}
    tangent=tangent_from_phase(oracle["query_phase"])
    for arm in ["original_lr","low_lr"]:
        for mode in ["last","average"]:
            arrays_path=args.run/f"{arm}_{mode}_arrays.npz";saved=np.load(arrays_path)
            np.testing.assert_array_equal(saved["queries"],oracle["queries"])
            np.testing.assert_array_equal(saved["query_ids"],oracle["query_ids"])
            np.testing.assert_allclose(saved["scales"],oracle["scales"],rtol=1e-9,atol=0)
            sources[str(arrays_path)]=sha256(arrays_path)
            scales=saved["scales"];queries=torch.from_numpy(saved["queries"]).double().to(args.device)
            field=load_refinement_field(checkpoint,arm,mode,args.device)
            result=decompose(saved["queries"],saved["posterior"],oracle["posterior"],saved["response"],oracle["response"],scales)
            components=value_components(result["posterior_error"],tangent,scales)
            accumulated={}
            for i,s in enumerate(scales):
                value,j=active_jacobian(lambda q:diffusion_denoiser_output(field,q,float(s),preconditioning="edm",data_std=1.),queries,active=30)
                value,j=value.cpu().numpy(),j.cpu().numpy()
                np.testing.assert_allclose(value,saved["posterior"][i],rtol=1e-9,atol=1e-9)
                np.testing.assert_allclose(np.trace(j,axis1=-2,axis2=-1),saved["response"][i],rtol=1e-9,atol=1e-9)
                active=j[:,:20,:20];truth=oracle["jacobian_active"][i]
                tangent_r=np.einsum("bi,bij,bj->b",tangent,active,tangent)
                tangent_truth=np.einsum("bi,bij,bj->b",tangent,truth,tangent)
                normal_r=np.trace(active,axis1=-2,axis2=-1)-tangent_r
                normal_truth=np.trace(truth,axis1=-2,axis2=-1)-tangent_truth
                data={"jacobian":j,"response_tangent":tangent_r,"response_curve_normal":normal_r,
                    "response_ambient_normal":np.trace(j[:,20:,20:],axis1=-2,axis2=-1),
                    "response_tangent_error":tangent_r-tangent_truth,
                    "response_curve_normal_error":normal_r-normal_truth,
                    "jacobian_ambient_normal_squared":(j[:,20:,20:]**2).sum((-1,-2)),
                    **split_jacobian_error(active-truth,tangent)}
                for key,v in data.items():accumulated.setdefault(key,[]).append(v)
                row={"arm":arm,"mode":mode,"lambda":float(s),"count":len(queries),
                    "full_d_mae":float(np.abs(result["full"][i]-1).mean()),
                    "full_oracle_mae":float(np.abs(result["full"][i]-result["oracle_full"][i]).mean()),
                    "response_oracle_mae":float(np.abs(result["response_error"][i]).mean()),
                    **{key+"_mean":float(v[i].mean()) for key,v in result.items() if v.ndim==2},
                    **{key+"_mean":float(v.mean()) for key,v in data.items() if v.ndim==1},
                    **{"posterior_error_"+key+"_energy":float(v[i].mean()) for key,v in components.items()}}
                rows.append(row);print(json.dumps(row),flush=True)
            np.savez_compressed(args.output/f"{arm}_{mode}.npz",scales=scales,query_ids=saved["query_ids"],
                tangent=tangent,**result,**{"posterior_error_"+k+"_energy":v for k,v in components.items()},
                **{k:np.stack(v) for k,v in accumulated.items()})
        temporal_path=args.run/f"{arm}_temporal.npz";t=np.load(temporal_path)
        sources[str(temporal_path)]=sha256(temporal_path)
        np.testing.assert_array_equal(t["query_ids"],oracle["query_ids"][:64])
        variation=t["posterior"]-t["mean_function_posterior"][None]
        parts=value_components(variation,tangent[:64],t["scales"])
        np.testing.assert_allclose(sum(parts.values()).mean(0),t["posterior_variation"],rtol=1e-11,atol=1e-11)
        for i,s in enumerate(t["scales"]):
            temporal_rows.append({"arm":arm,"lambda":float(s),
                "mean_snapshot_correction":float(t["correction_original"][:,i].mean()),
                "mean_function_correction":float(t["mean_function_correction"][i].mean()),
                "posterior_variation":float(t["posterior_variation"][i].mean()),
                **{"variation_"+key:float(v[:,i].mean()) for key,v in parts.items()}})
        np.savez_compressed(args.output/f"{arm}_temporal_components.npz",scales=t["scales"],query_ids=t["query_ids"],
            **{"variation_"+k:v for k,v in parts.items()})
    for name,data in [("summary",rows),("temporal",temporal_rows)]:
        (args.output/f"{name}.json").write_text(json.dumps(data,indent=2)+"\n")
        with (args.output/f"{name}.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    assert sha256(checkpoint)==before
    (args.output/"manifest.json").write_text(json.dumps({"source_sha256":sha256(__file__),
        "checkpoint":str(checkpoint),"checkpoint_sha256":before,"oracle":str(args.oracle),"oracle_sha256":sha256(args.oracle),
        "sources":sources,"geometry":"Oracle 1D tangent;19 within-span curve normals;10 ambient normals",
        "checks":"Exact full30 spatial Jacobian agrees with stored R and b; temporal orthogonal decomposition and signed C identities",
        "interpretation":"Paired fresh-population target-noise removal; oracle targets are nondeployable; no model/scale selection",
        "source_checkpoint_unchanged":True},indent=2)+"\n")
    return {"rows":len(rows),"output":str(args.output)}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,default=Path("artifacts/response_full_20260907/fixed_objective_lr_spaghetti"))
    p.add_argument("--oracle",type=Path,default=Path("artifacts/response_full_20260907/population_oracle_ve_spaghetti256/arrays.npz"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda:0")
    print(json.dumps(run(p.parse_args())),flush=True)
