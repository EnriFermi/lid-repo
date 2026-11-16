"""Finite normal-ray posterior-invariance diagnostic at one prespecified scale.

No training, no residual injection and no parameter gradients. The affine
population posterior is identical along each normal ray; all departures are
properties of the frozen learned posterior function.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from experiments.fm_mechanism_condition_refit import make_oracle
from experiments.fm_scheduler_cause_training import (ANCHOR,ARM_SCHEDULE,
    arm_posterior,load_cause_checkpoint)


RHOS=(0.,.25,.5,.75,1.)


def normal_ray_inputs(observed,basis,center,rhos=RHOS):
    basis=torch.linalg.qr(basis.to(observed)).Q
    center=center.to(observed)
    tangent=center+((observed-center)@basis)@basis.T
    normal=observed-tangent
    return basis,tangent,normal,{rho:tangent+rho*normal for rho in rhos}


def error_components(prediction,truth,baseline,basis,tangent_centered):
    error=prediction-truth
    projected=error@basis
    energy=error.square().sum(-1)
    tangent_energy=projected.square().sum(-1)
    delta=prediction-baseline
    delta_tangent=delta@basis
    return {"tangent_error":projected,"tangent_energy":tangent_energy,
        "normal_energy":(energy-tangent_energy).clamp_min(0),
        "delta_tangent":delta_tangent,
        "delta_normal_energy":(delta.square().sum(-1)-delta_tangent.square().sum(-1)).clamp_min(0),
        "radial_error_numerator":(error*tangent_centered).sum(-1),
        "radial_error_denominator":tangent_centered.square().sum(-1)}


@torch.no_grad()
def run(args):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True,exist_ok=False)
    rows,raw,manifests=[],{},[]
    for run_path in args.runs:
        source_manifest=json.loads((run_path/"manifest.json").read_text())
        fixture=torch.load(run_path/"data.pt",map_location="cpu",weights_only=False)
        if fixture["data_law"] not in ("gaussian","uniform"):
            raise ValueError("Exact normal-invariance probe requires an affine law")
        data_path=run_path/"evaluation_data.pt"
        data=torch.load(data_path,map_location="cpu",weights_only=False)
        if args.count>len(data["clean"]):raise ValueError("Not enough preserved heldout examples")
        clean,noise=data["clean"][:args.count].double(),data["noise"][:args.count].double()
        observed=clean+args.scale*noise
        basis,tangent,normal,rays=normal_ray_inputs(observed,fixture["tangent_basis"],fixture["center"])
        oracle,_=make_oracle(source_manifest["effective_config"],fixture,"cpu")
        truth=oracle(tangent,args.scale).mean
        invariance_errors={rho:float((oracle(points,args.scale).mean-truth).abs().max()) for rho,points in rays.items()}
        if max(invariance_errors.values())>1e-9:raise AssertionError(invariance_errors)
        record={"run":str(run_path),"data_law":fixture["data_law"],
            "evaluation_data_sha256":hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "oracle_invariance_max_absolute":invariance_errors,"checkpoints":{}}
        raw[run_path.name]={}
        for arm in ARM_SCHEDULE:
            checkpoint=run_path/(arm+"_tailavg")/"checkpoint_step_009000.pt"
            digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            field,payload=load_cause_checkpoint(checkpoint,"cpu")
            field=field.double().eval()
            baseline=None
            raw[run_path.name][arm]={}
            for rho,points in rays.items():
                prediction=torch.cat([arm_posterior(field,p,args.scale,arm,args.anchor) for p in points.split(256)])
                if baseline is None:baseline=prediction
                components=error_components(prediction,truth,baseline,basis,tangent-fixture["center"].double())
                raw[run_path.name][arm][rho]=components
                units=points.shape[-1]*args.scale**2
                rows.append({"run":run_path.name,"law":fixture["data_law"],"arm":arm,"lambda":args.scale,
                    "rho":rho,"count":args.count,"tangent_excess_risk":float(components["tangent_energy"].mean()/units),
                    "normal_excess_risk":float(components["normal_energy"].mean()/units),
                    "total_excess_risk":float((components["tangent_energy"]+components["normal_energy"]).mean()/units),
                    "normal_ray_tangent_change_energy_over_Dlambda2":float(components["delta_tangent"].square().sum(-1).mean()/units),
                    "normal_ray_normal_change_energy_over_Dlambda2":float(components["delta_normal_energy"].mean()/units),
                    "radial_gain_error":float(components["radial_error_numerator"].sum()/components["radial_error_denominator"].sum()),
                    "mean_normal_input_norm":float(rho*normal.norm(dim=-1).mean())})
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
            record["checkpoints"][arm]={"path":str(checkpoint),"sha256":digest,
                "output_coordinate":payload["config"]["output_coordinate"],
                "output_normalization":payload["config"].get("output_normalization","layernorm")}
        manifests.append(record)
    manifest={"scale":args.scale,"anchor":args.anchor,"rhos":RHOS,"count":args.count,"device":"cpu",
        "dtype":"float64","threads":4,"query_protocol":"All first count saved heldout examples; tangent of original noisy observation held fixed; no query selection",
        "comparison":"Finite ambient-normal displacements; oracle posterior invariant; no parameter or gradient intervention",
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"runs":manifests,
        "raw_note":"Stores per-query tangent errors and normal energies, not every ambient prediction vector; source checkpoints permit exact replay"}
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (args.output/"summary.json").write_text(json.dumps(rows,indent=2)+"\n")
    with (args.output/"summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    torch.save(raw,args.output/"query_components.pt")
    print(json.dumps({"runs":len(args.runs),"rows":len(rows),"output":str(args.output),"all_sources_unchanged":True}))


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument("--runs",type=Path,nargs="+",required=True)
    result.add_argument("--output",type=Path,required=True)
    result.add_argument("--scale",type=float,default=ANCHOR)
    result.add_argument("--anchor",type=float,default=ANCHOR)
    result.add_argument("--count",type=int,default=2048)
    return result


if __name__=="__main__":run(parser().parse_args())
