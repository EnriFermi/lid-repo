"""Fixed-recipe VE refinement transfer to the actual Exp narrowing surface."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
from experiments.response_full_models_20260907_span import sha256
from experiments.response_full_models_20260907_spaghetti_local import active_jacobian
from experiments.response_full_models_20260907_oracle_audit import decompose
from experiments.response_full_models_20260907_ve_temporal import temporal_trend
from experiments.response_full_models_20260907_ve_refinement import load_refinement_field
from experiments.response_full_oracle_20260907_geometry import local_frame
from models.training import diffusion_denoiser_output


def matrix_components(error,normal):
    n=normal[:,:,None];pn=n@n.transpose(0,2,1);pt=np.eye(error.shape[-1])[None]-pn
    result={"tangent_tangent":pt@error@pt,"tangent_normal":pt@error@pn,
        "normal_tangent":pn@error@pt,"normal_normal":pn@error@pn}
    np.testing.assert_allclose(sum((v**2).sum((-1,-2)) for v in result.values()),
        (error**2).sum((-1,-2)),rtol=1e-11,atol=1e-11)
    return result


def vector_components(error,normal,scales):
    s2=np.asarray(scales)[:,None]**2
    normal_energy=np.einsum("...qd,qd->...q",error[...,:3],normal)**2/s2
    tangent=(error[...,:3]**2).sum(-1)/s2-normal_energy
    ambient=(error[...,3:]**2).sum(-1)/s2
    return {"tangent":tangent,"surface_normal":normal_energy,"ambient_normal":ambient}


def run(args):
    torch.set_num_threads(4);args.output.mkdir(parents=True,exist_ok=False)
    oracle=np.load(args.oracle);jac_oracle=np.load(args.oracle_jacobian)
    for key in ["query_ids","queries_raw","scales","raw_sigma","response","correction"]:
        np.testing.assert_array_equal(oracle[key],jac_oracle[key])
    normal=local_frame(oracle["queries_raw"][:,:3],"tube")
    checkpoint=args.run/"final.pt";before=sha256(checkpoint)
    rows,temporal_rows,sources=[],[],{}
    for arm in ["original_lr","low_lr"]:
        for mode in ["last","average"]:
            source=args.run/f"{arm}_{mode}_arrays.npz";a=np.load(source);sources[str(source)]=sha256(source)
            np.testing.assert_array_equal(a["queries"],oracle["queries"])
            np.testing.assert_array_equal(a["query_ids"],oracle["query_ids"])
            np.testing.assert_allclose(a["scales"],oracle["scales"],rtol=1e-9,atol=0)
            scales=a["scales"];q=torch.from_numpy(a["queries"]).double().to(args.device)
            field=load_refinement_field(checkpoint,arm,mode,args.device)
            result=decompose(a["queries"],a["posterior"],oracle["posterior"],a["response"],oracle["response"],scales)
            components=vector_components(result["posterior_error"],normal,scales);stored={}
            for i,s in enumerate(scales):
                b,j=active_jacobian(lambda x:diffusion_denoiser_output(field,x,float(s),preconditioning="edm",data_std=1.),q,active=30)
                b,j=b.cpu().numpy(),j.cpu().numpy()
                np.testing.assert_allclose(b,a["posterior"][i],rtol=1e-9,atol=1e-9)
                np.testing.assert_allclose(np.trace(j,axis1=-2,axis2=-1),a["response"][i],rtol=1e-9,atol=1e-9)
                active=j[:,:3,:3];truth=jac_oracle["jacobian_active"][i]
                nr=np.einsum("bi,bij,bj->b",normal,active,normal)
                nstar=np.einsum("bi,bij,bj->b",normal,truth,normal)
                tr=np.trace(active,axis1=-2,axis2=-1)-nr
                tstar=np.trace(truth,axis1=-2,axis2=-1)-nstar
                jparts=matrix_components(active-truth,normal)
                data={"jacobian":j,"response_surface_normal":nr,"response_tangent":tr,
                    "response_ambient_normal":np.trace(j[:,3:,3:],axis1=-2,axis2=-1),
                    "response_surface_normal_error":nr-nstar,"response_tangent_error":tr-tstar,
                    "jacobian_ambient_normal_squared":(j[:,3:,3:]**2).sum((-1,-2)),
                    **{"jacobian_error_"+k+"_squared":(v**2).sum((-1,-2)) for k,v in jparts.items()}}
                for k,v in data.items():stored.setdefault(k,[]).append(v)
                row={"arm":arm,"mode":mode,"lambda":float(s),"count":len(q),
                    "full_d_mae":float(np.abs(result["full"][i]-2).mean()),
                    "response_d_mae":float(np.abs(a["response"][i]-2).mean()),
                    "full_oracle_mae":float(np.abs(result["full"][i]-result["oracle_full"][i]).mean()),
                    "response_oracle_mae":float(np.abs(result["response_error"][i]).mean()),
                    **{k+"_mean":float(v[i].mean()) for k,v in result.items() if v.ndim==2},
                    **{k+"_mean":float(v.mean()) for k,v in data.items() if v.ndim==1},
                    **{"posterior_error_"+k+"_energy":float(v[i].mean()) for k,v in components.items()}}
                rows.append(row);print(json.dumps(row),flush=True)
            np.savez_compressed(args.output/f"{arm}_{mode}.npz",query_ids=a["query_ids"],scales=scales,normal=normal,
                **result,**{"posterior_error_"+k+"_energy":v for k,v in components.items()},
                **{k:np.stack(v) for k,v in stored.items()})
        source=args.run/f"{arm}_temporal.npz";a=np.load(source);sources[str(source)]=sha256(source)
        np.testing.assert_array_equal(a["query_ids"],oracle["query_ids"][:64])
        ids=[int(np.argmin(np.abs(oracle["scales"]-s))) for s in a["scales"]]
        np.testing.assert_allclose(a["scales"],oracle["scales"][ids],rtol=1e-9,atol=0)
        scales=a["scales"]
        result=decompose(oracle["queries"][:64],a["mean_function_posterior"],oracle["posterior"][ids,:64],
            a["mean_function_response"],oracle["response"][ids,:64],scales)
        trend=temporal_trend(a["posterior"],a["steps"].copy(),scales)
        components=vector_components(a["posterior"]-a["mean_function_posterior"][None],normal[:64],scales)
        np.testing.assert_allclose(sum(components.values()).mean(0),trend["total_variation"],rtol=1e-11,atol=1e-11)
        mean_full=a["full"].mean(0)
        np.testing.assert_allclose(mean_full,result["full"]+trend["total_variation"],rtol=1e-10,atol=1e-10)
        for i,s in enumerate(scales):
            total=float(trend["total_variation"][i].mean())
            temporal_rows.append({"arm":arm,"lambda":float(s),"snapshot_correction":float(a["correction_original"][:,i].mean()),
                "mean_of_full_d_mae":float(np.abs(mean_full[i]-2).mean()),
                "full_of_mean_d_mae":float(np.abs(result["full"][i]-2).mean()),
                "mean_of_full_oracle_mae":float(np.abs(mean_full[i]-result["oracle_full"][i]).mean()),
                "full_of_mean_oracle_mae":float(np.abs(result["full"][i]-result["oracle_full"][i]).mean()),
                **{k+"_mean":float(v[i].mean()) for k,v in result.items() if v.ndim==2},
                **{k:float(v[i].mean()) for k,v in trend.items()},
                **{"variation_"+k:float(v[:,i].mean()) for k,v in components.items()},
                "linear_trend_fraction":float(trend["linear_trend_variation"][i].mean()/total)})
        np.savez_compressed(args.output/f"{arm}_temporal.npz",query_ids=a["query_ids"],scales=scales,
            mean_of_full=mean_full,**result,**trend,**{"variation_"+k:v for k,v in components.items()})
    for name,data in [("summary",rows),("temporal",temporal_rows)]:
        (args.output/f"{name}.json").write_text(json.dumps(data,indent=2)+"\n")
        with (args.output/f"{name}.csv").open("w",newline="") as stream:
            w=csv.DictWriter(stream,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    assert sha256(checkpoint)==before
    (args.output/"manifest.json").write_text(json.dumps({"source_sha256":sha256(__file__),"checkpoint_sha256":before,
        "oracle_sha256":sha256(args.oracle),"oracle_jacobian_sha256":sha256(args.oracle_jacobian),"sources":sources,
        "geometry":"Exact rank2 tangent of Exp surface;1 surface normal;27 ambient normals; diagnostics only",
        "temporal_scope":"16 sparse fixed snapshots: total response exact, no matrix blocks saved; linear trend is a descriptive control",
        "population":"Same original uniform-u surface, not replaced by cylinder or area-uniform law",
        "source_checkpoint_unchanged":True},indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,default=Path("artifacts/response_full_20260907/ve_exp_refinement"))
    p.add_argument("--oracle",type=Path,default=Path("artifacts/response_full_20260907/population_oracle_ve_exp256/fixture_arrays.npz"))
    p.add_argument("--oracle-jacobian",type=Path,default=Path("artifacts/response_full_20260907/oracle_eigen_gate/tube_arrays.npz"))
    p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda:0")
    run(p.parse_args())
