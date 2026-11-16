"""Exact tangent-output Jacobian blocks and finite-noise error accounting.

Uses d reverse-mode passes, not D full-Jacobian passes. Linearized Gaussian
noise risks are explicitly distinguished from finite-scale network errors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from experiments.fm_scheduler_cause_training import ARM_SCHEDULE, arm_posterior, load_cause_checkpoint, write_csv


def tangent_jacobian(predict, points, basis):
    x = points.detach().requires_grad_(True)
    value = predict(x)
    tangent_value = value @ basis
    rows = [torch.autograd.grad(tangent_value[:, j].sum(), x, retain_graph=True)[0]
            for j in range(basis.shape[1])]
    ambient = torch.stack(rows, dim=1)
    tt = ambient @ basis
    tn = ambient - tt @ basis.T
    return value.detach(), ambient.detach(), tt.detach(), tn.detach()


def analyze(predict, points, noise, basis, scale, oracle_gain):
    value, jacobian, tt, tn = tangent_jacobian(predict, points, basis)
    d = basis.shape[1]
    identity = torch.eye(d, device=points.device, dtype=points.dtype)
    tt_error = tt - oracle_gain * identity
    qt = points @ basis
    noise_tangent = (noise @ basis) @ basis.T
    noise_normal = noise-noise_tangent
    error_clean = value @ basis - oracle_gain * qt
    linear_increment = scale * (torch.einsum("bid,bd->bi", jacobian, noise)-oracle_gain*(noise@basis))
    linear_error = error_clean+linear_increment
    with torch.no_grad():
        noisy_value = predict(points+scale*noise)
        noisy_error = noisy_value@basis-oracle_gain*((points+scale*noise)@basis)
        normal_positive = predict(points+scale*noise_normal)@basis-value@basis
        normal_negative = predict(points-scale*noise_normal)@basis-value@basis
        normal_even = (normal_positive+normal_negative)/2
        normal_odd = (normal_positive-normal_negative)/2
        tangent_error = predict(points+scale*noise_tangent)@basis-oracle_gain*((points+scale*noise_tangent)@basis)
    remainder = noisy_error-linear_error
    energy = lambda x: float(x.square().sum(-1).mean()/scale**2)
    clean_energy = energy(error_clean)
    tt_energy = float(tt_error.square().sum((-1,-2)).mean())
    tn_energy = float(tn.square().sum((-1,-2)).mean())
    result = {"clean_tangent_bias_over_lambda2":clean_energy,
              "tt_oracle_frobenius_squared":tt_energy,
              "tn_frobenius_squared":tn_energy,
              "expected_linearized_tangent_risk":clean_energy+tt_energy+tn_energy,
              "realized_linearized_tangent_risk":energy(linear_error),
              "actual_noisy_tangent_risk":energy(noisy_error),
              "linearization_remainder_energy":energy(remainder),
              "linearization_twice_cross":float(2*(linear_error*remainder).sum(-1).mean()/scale**2),
              "normal_only_positive_tangent_risk":energy(error_clean+normal_positive),
              "normal_only_displacement_even_energy":energy(normal_even),
              "normal_only_displacement_odd_energy":energy(normal_odd),
              "normal_only_even_radial_gain":float((normal_even*qt).sum()/qt.square().sum()),
              "tangent_only_tangent_risk":energy(tangent_error),
              "mean_tangent_trace":float(tt.diagonal(dim1=-2,dim2=-1).sum(-1).mean())}
    result["energy_identity_error"] = abs(result["actual_noisy_tangent_risk"]-result["realized_linearized_tangent_risk"]-result["linearization_remainder_energy"]-result["linearization_twice_cross"])
    arrays = dict(points=points,noise=noise,value=value,jacobian_tangent_output=jacobian,
                  tt=tt,tn=tn,error_clean_tangent=error_clean,noisy_error_tangent=noisy_error,
                  linear_error_tangent=linear_error,linearization_remainder_tangent=remainder,
                  normal_displacement_even=normal_even,normal_displacement_odd=normal_odd)
    return result, {k:v.detach().cpu() for k,v in arrays.items()}


def run(source_roots, output, *, device="cpu", count=64, scale=.316227766):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    rows,arrays,sources=[],{},{}
    start=time.monotonic()
    for root in map(Path,source_roots):
        fixture=torch.load(root/"data.pt",map_location="cpu",weights_only=False)
        if fixture["data_law"]!="gaussian":raise ValueError("Gaussian affine oracle required")
        basis=torch.linalg.qr(fixture["tangent_basis"].double()).Q.to(device)
        tau2=float(fixture["latent_std"])**2
        oracle_gain=tau2/(tau2+scale**2)
        evaluation=torch.load(root/"evaluation_data.pt",map_location="cpu",weights_only=False)
        noise=evaluation["noise"][:count].to(device=device,dtype=torch.float64)
        for arm in ARM_SCHEDULE:
            checkpoint=root/f"{arm}_tailavg"/"checkpoint_step_009000.pt"
            digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            field,payload=load_cause_checkpoint(checkpoint,device)
            field=field.double().eval()
            for parameter in field.parameters():parameter.requires_grad_(False)
            def predict(x):return arm_posterior(field,x,x.new_full((len(x),),scale),arm,payload["anchor"])
            sources[f"{root.name}/{arm}"]={"path":str(checkpoint),"sha256":digest,
                "output_coordinate":field.pilot_output_coordinate,"normalization":payload["config"].get("output_normalization","layernorm")}
            for group in ("queries","clean"):
                points=evaluation[group][:count].to(device=device,dtype=torch.float64)
                metrics,values=analyze(predict,points,noise,basis,scale,oracle_gain)
                rows.append({"source":root.name,"arm":arm,"group":group,"count":len(points),"lambda":scale,"oracle_gain":oracle_gain,**metrics})
                arrays.update({f"{root.name}/{arm}/{group}/{k}":v for k,v in values.items()})
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=digest:raise AssertionError("Source changed")
            print(json.dumps({"source":root.name,"arm":arm,"elapsed_seconds":time.monotonic()-start}),flush=True)
    write_csv(output/"metrics.csv",rows)
    torch.save(arrays,output/"arrays.pt")
    summary={"sources":sources,"count":count,"scale":scale,"elapsed_seconds":time.monotonic()-start,
             "maximum_energy_identity_error":max(r["energy_identity_error"] for r in rows),
             "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "scope":"Exact local TT/TN blocks of canonical posterior; input/scale labels and TOTAL native output normalization preserved. Expected risk is linearized, not exact finite-scale risk. Normal-only +/- noise splits finite displacement into even/odd components. All risks are total tangent energy/lambda^2, not divided by ambient D.",
             "groups":"queries is the saved clean LID-query subset; clean is the saved clean base for the first matching noisy-risk examples."}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-roots",nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--count",type=int,default=64)
    args=parser.parse_args()
    print(json.dumps(run(**vars(args)),indent=2),flush=True)


if __name__=="__main__":main()
