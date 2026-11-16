"""Controlled Gaussian probability-flow-form transport from exact p_1 to .01.

No training and no Gaussian-prior shortcut: initialization is fresh population
X+Z in canonical coordinates. For exact b the ODE dr/dlog(lambda)=r-b has the
Gaussian-smoothed marginals. A learned b defines an approximate transport.
The coupled X+.01Z reference has the correct marginal but is not a samplewise
ground-truth endpoint map.
"""
from __future__ import annotations
import argparse,csv,json,math,time
from pathlib import Path
import numpy as np
import torch
from experiments.response_full_models_20260907_span import sha256
from experiments.response_full_models_20260907_vp import load_field
from experiments.response_full_multiscale_20260907 import project
from experiments.response_full_training_20260907 import posterior
from experiments.ve_known_lid_recoverability_campaign import sample_benchmark_coefficients


def interpolate_gamma(scale,grid,values):
    if scale<grid[0]*(1-1e-12) or scale>grid[-1]*(1+1e-12):
        raise ValueError("No extrapolation outside frozen calibration support")
    return float(np.interp(math.log(scale),np.log(grid),values))


def calibrated_projected(b,x,center,basis,gamma):
    anchor=project(x,center,basis)
    return anchor+gamma*(project(b,center,basis)-anchor)


@torch.no_grad()
def integrate(predict,initial,*,steps=128,start=1.,end=.01,record=True):
    """Classical RK4, decreasing log-scale; no clipping or geometry-dependent step."""
    if not 0<end<start or steps<1:raise ValueError("Invalid transport interval")
    times=np.linspace(math.log(start),math.log(end),steps+1)
    x=initial.clone();snapshots=[x.clone()] if record else []
    saved_scales=[start] if record else []
    for i in range(steps):
        t=float(times[i]);h=float(times[i+1]-times[i])
        scale1=math.exp(t);scale2=math.exp(t+h/2);scale3=math.exp(t+h)
        k1=x-predict(x,scale1)
        k2=x+h*k1/2-predict(x+h*k1/2,scale2)
        k3=x+h*k2/2-predict(x+h*k2/2,scale2)
        k4=x+h*k3-predict(x+h*k3,scale3)
        x=x+h*(k1+2*k2+2*k3+k4)/6
        if not torch.isfinite(x).all():raise FloatingPointError(f"Nonfinite endpoint atstep{i+1}")
        if record and (i+1)*4%steps==0:
            snapshots.append(x.clone());saved_scales.append(scale3)
    return x,torch.stack(snapshots) if record else None,np.asarray(saved_scales)


def prepare_fixture(dataset,source_fixture,*,count=1024,data_seed=20261207,noise_seed=20261208):
    raw=sample_benchmark_coefficients(dataset,count,seed=data_seed)
    mean=source_fixture["mean"].double().cpu();rms=float(source_fixture["rms"])
    x=(torch.from_numpy(raw).double()-mean)/rms
    z=torch.randn(x.shape,dtype=torch.float64,generator=torch.Generator().manual_seed(noise_seed))
    scales=torch.exp(torch.linspace(0,math.log(.01),5,dtype=torch.float64))
    reference=x[None]+scales[:,None,None]*z[None]
    return {"clean_raw":raw,"clean_canonical":x.numpy(),"noise":z.numpy(),"initial":(x+z).numpy(),
        "reference":(x+.01*z).numpy(),"reference_trajectory":reference.numpy(),
        "reference_raw":((x+.01*z)*rms+mean).numpy(),"normalization_mean":mean.numpy(),
        "normalization_rms":np.asarray(rms),"reference_scales":scales.numpy(),"sample_ids":np.arange(count)}


def run(args):
    torch.set_num_threads(4);args.output.mkdir(parents=True,exist_ok=False)
    root=Path("artifacts/response_full_20260907");all_rows=[]
    for name in args.datasets:
        start_time=time.monotonic();out=args.output/name;out.mkdir()
        dataset={"spaghetti":"e8_spaghetti_pca","exp":"e6_exp_pca"}[name]
        checkpoint=root/f"vp_{name}_paired/final.pt";fixture_path=checkpoint.parent/"fixture.pt"
        calibration=root/f"calibration_vp_{name}_posterior/arrays.npz"
        base_path=root/f"vp_{name}_span/posterior_average_arrays.npz"
        paths=[checkpoint,fixture_path,calibration,base_path,Path(__file__),
            Path("experiments/response_full_training_20260907.py"),Path("models/neural_fields.py")]
        hashes={str(p):sha256(p) for p in paths}
        payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
        source_fixture=torch.load(fixture_path,map_location="cpu",weights_only=False)
        fixture=prepare_fixture(dataset,source_fixture,count=args.count)
        a=np.load(base_path);cal=np.load(calibration)
        np.testing.assert_array_equal(a["normalization_mean"],source_fixture["mean"].numpy())
        assert float(a["normalization_rms"])==float(source_fixture["rms"])
        np.testing.assert_array_equal(a["scales"],cal["scales"])
        fixture.update(span_center=a["span_center"],span_basis=a["span_basis"],
            calibration_scales=cal["scales"],calibration_gamma=cal["projected_gamma"])
        np.savez_compressed(out/"fixture.npz",**fixture)
        manifest={"source_hashes":hashes,"dataset":dataset,"count":args.count,"initial_lambda":1.,"endpoint_lambda":.01,
            "model_modes":"Fixed last1000 parameter averages from original paired VP runs",
            "ode":"dr/dlog(lambda)=r-b; RK4 fromlog1 tolog.01",
            "initialization":"Exact population canonicalX+Z, NOT standardGaussian prior",
            "reference":"SameX+.01Z gives correctp_.01 marginal, not samplewise target coupling",
            "data_seed":20261207,"noise_seed":20261208,"steps":args.steps,"convergence_steps":2*args.steps,
            "convergence_count":args.convergence_count,"dtype":"float64",
            "variants":["posterior","projected_posterior","calibrated_projected_posterior","noise"],
            "calibration":"Frozen projected_gamma values, linear interpolation inloglambda; no retuning or extrapolation",
            "numerical_gate":"First128 or declared convergence_count: meanendpointdifference/(.01sqrtD)<=.02; max<=.1",
            "geometry_scope":"Closest-continuous-support distance AND latentcoverage; oracle geometry never enters velocity"}
        (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        fields={k:load_field(payload,k,"average",args.device) for k in ["posterior","noise"]}
        center=torch.from_numpy(a["span_center"]).double().to(args.device)
        basis=torch.from_numpy(a["span_basis"]).double().to(args.device)
        initial=torch.from_numpy(fixture["initial"]).double().to(args.device)
        queries=torch.from_numpy(a["queries"]).double().to(args.device)
        with torch.no_grad():
            for i,s in enumerate(a["scales"]):
                b=posterior(fields["posterior"],queries,float(s),"posterior")
                np.testing.assert_allclose(b.cpu().numpy(),a["posterior"][i],rtol=1e-9,atol=1e-9)
                g=calibrated_projected(b,queries,center,basis,float(cal["projected_gamma"][i]))
                np.testing.assert_allclose(g.cpu().numpy(),cal["projected_posterior"][i],rtol=1e-9,atol=1e-9)
        outputs={};rows=[]
        for variant in manifest["variants"]:
            kind="noise" if variant=="noise" else "posterior"
            def predict(x,scale):
                b=posterior(fields[kind],x,scale,kind)
                if variant=="projected_posterior":return project(b,center,basis)
                if variant=="calibrated_projected_posterior":
                    gamma=interpolate_gamma(scale,cal["scales"],cal["projected_gamma"])
                    return calibrated_projected(b,x,center,basis,gamma)
                return b
            tick=time.monotonic()
            endpoint,trajectory,trajectory_scales=integrate(predict,initial,steps=args.steps)
            refined,_,_=integrate(predict,initial[:args.convergence_count],steps=2*args.steps,record=False)
            difference=(endpoint[:args.convergence_count]-refined).norm(dim=1)/(.01*math.sqrt(initial.shape[1]))
            passed=float(difference.mean())<=.02 and float(difference.max())<=.1
            row={"dataset":dataset,"variant":variant,"count":args.count,
                "elapsed_seconds":time.monotonic()-tick,"mean_refinement_difference_over_tube_rms":float(difference.mean()),
                "maximum_refinement_difference_over_tube_rms":float(difference.max()),"numerical_gate_passed":passed}
            outputs[variant+"_endpoint"]=endpoint.cpu().numpy()
            outputs[variant+"_endpoint_raw"]=outputs[variant+"_endpoint"]*float(fixture["normalization_rms"])+fixture["normalization_mean"]
            outputs[variant+"_trajectory"]=trajectory.cpu().numpy()
            outputs[variant+"_refined_subset"]=refined.cpu().numpy()
            outputs[variant+"_refinement_difference_over_tube_rms"]=difference.cpu().numpy()
            rows.append(row);print(json.dumps(row),flush=True)
            np.savez_compressed(out/"endpoints.npz",trajectory_scales=trajectory_scales,**outputs)
        assert all(sha256(p)==h for p,h in hashes.items())
        (out/"summary.json").write_text(json.dumps(rows,indent=2)+"\n")
        (out/"status.json").write_text(json.dumps({"status":"complete","all_numerical_gates_passed":all(r["numerical_gate_passed"] for r in rows),
            "all_sources_unchanged":True,"elapsed_seconds":time.monotonic()-start_time})+"\n")
        all_rows.extend(rows)
    (args.output/"summary.json").write_text(json.dumps(all_rows,indent=2)+"\n")
    with (args.output/"summary.csv").open("w",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--datasets",nargs="+",choices=["spaghetti","exp"],default=["spaghetti","exp"])
    p.add_argument("--count",type=int,default=1024);p.add_argument("--steps",type=int,default=128)
    p.add_argument("--convergence-count",type=int,default=128);p.add_argument("--device",default="cuda:0")
    run(p.parse_args())
