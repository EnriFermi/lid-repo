"""Fixed-local-frame Jacobian audit of the product-circle models.

Only the network is differentiated. Local tangent/radial frames are computed
from detached queries and remain fixed in each derivative. Forty span-output
VJPs recover both traces; the remaining ambient trace uses the saved full trace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from experiments.fm_mechanism_circle_oracle import circle_posterior
from experiments.fm_scheduler_cause_jacobian_blocks import tangent_jacobian
from experiments.fm_scheduler_cause_torus import LAW, posterior
from experiments.fm_scheduler_cause_training import ARM_SCHEDULE, arm_posterior, load_cause_checkpoint, write_csv


def local_rotation(points, fixture):
    """Columns: d local tangents, then d circle-radial normals, in span coordinates."""
    detached=points.detach()
    basis=fixture["embedding_basis"].to(detached)
    pairs=((detached-fixture["center"].to(detached))@basis).reshape(len(points),-1,2)
    radius=pairs.norm(dim=-1,keepdim=True)
    if bool((radius<=0).any()):raise ValueError("Undefined local frame at a circle-plane origin")
    radial=pairs/radius
    tangent=torch.stack((-radial[...,1],radial[...,0]),dim=-1)
    d=fixture["circle_count"]
    rotation=points.new_zeros(len(points),2*d,2*d)
    for j in range(d):
        rotation[:,2*j:2*j+2,j]=tangent[:,j]
        rotation[:,2*j:2*j+2,d+j]=radial[:,j]
    return rotation.detach(),pairs.detach()


def analyze(predict, points, fixture, scale, total_trace=None):
    basis=fixture["embedding_basis"].to(points)
    rotation,pairs=local_rotation(points,fixture)
    d=fixture["circle_count"]
    value,ambient,span,_=tangent_jacobian(predict,points,basis)
    local=rotation.transpose(1,2)@span@rotation
    tt,rr=local[:,:d,:d],local[:,d:,d:]
    tr,rt=local[:,:d,d:],local[:,d:,:d]
    truth=posterior(points,scale,fixture)
    circles=circle_posterior(pairs,scale,radius=fixture["circle_radius"])
    oracle_tt=torch.diag_embed(circles.tangent_gain)
    oracle_rr=torch.diag_embed(circles.radial_gain)
    trace_t=tt.diagonal(dim1=-2,dim2=-1).sum(-1)
    trace_r=rr.diagonal(dim1=-2,dim2=-1).sum(-1)
    trace_span=span.diagonal(dim1=-2,dim2=-1).sum(-1)
    error=value-truth.mean
    error_span=error@basis
    error_local=torch.einsum("bi,bij->bj",error_span,rotation)
    normal_error=error-error_span@basis.T
    mean_energy=lambda x:float(x.square().flatten(1).sum(-1).mean())
    metrics={"tangent_trace":float(trace_t.mean()),"circle_radial_trace":float(trace_r.mean()),
             "span_trace":float(trace_span.mean()),
             "ambient_complement_trace":float((total_trace-trace_span).mean()) if total_trace is not None else None,
             "oracle_tangent_trace":float(circles.tangent_gain.sum(-1).mean()),
             "oracle_circle_radial_trace":float(circles.radial_gain.sum(-1).mean()),
             "tangent_block_error_squared":mean_energy(tt-oracle_tt),
             "circle_radial_block_error_squared":mean_energy(rr-oracle_rr),
             "radial_input_tangent_output_squared":mean_energy(tr),
             "tangent_input_radial_output_squared":mean_energy(rt),
             "span_identity_error_squared":mean_energy(span-torch.eye(2*d,device=points.device,dtype=points.dtype)),
             "tangent_value_error_over_lambda2":mean_energy(error_local[:,:d])/scale**2,
             "circle_radial_value_error_over_lambda2":mean_energy(error_local[:,d:])/scale**2,
             "ambient_complement_value_error_over_lambda2":mean_energy(normal_error)/scale**2,
             "trace_partition_max_error":float((trace_span-trace_t-trace_r).abs().max()),
             "oracle_full_mean":float(truth.full.mean())}
    arrays={"points":points,"value":value,"rotation":rotation,"span_jacobian":span,
            "local_jacobian":local,"tangent_trace":trace_t,"circle_radial_trace":trace_r,
            "span_trace":trace_span,"oracle_tangent_gain":circles.tangent_gain,
            "oracle_radial_gain":circles.radial_gain,"error_local":error_local,
            "error_ambient_complement":normal_error}
    if total_trace is not None:arrays["saved_total_trace"]=total_trace
    return metrics,{k:v.detach().cpu() for k,v in arrays.items()}


def run(source_roots,output,*,device="cpu",count=64,scale=.316227766):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows,arrays,sources=[],{},{}
    start=time.monotonic()
    for root in map(Path,source_roots):
        fixture=torch.load(root/"data.pt",map_location="cpu",weights_only=False)
        if fixture["data_law"]!=LAW:raise ValueError("Product-circle fixture required")
        data=torch.load(root/"evaluation_data.pt",map_location="cpu",weights_only=False)
        points=data["queries"][:count].to(device=device,dtype=torch.float64)
        for arm in ARM_SCHEDULE:
            directory=root/f"{arm}_tailavg"
            checkpoint=directory/"checkpoint_step_009000.pt"
            digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            field,payload=load_cause_checkpoint(checkpoint,device)
            field=field.double().eval()
            for parameter in field.parameters():parameter.requires_grad_(False)
            def predict(x):return arm_posterior(field,x,x.new_full((len(x),),scale),arm,payload["anchor"])
            saved=torch.load(directory/"evaluation_step_009000.pt",map_location=device,weights_only=False)[scale]
            metrics,values=analyze(predict,points,fixture,scale,saved["response"][:count])
            saved_error=float((values["value"]-saved["values"][:count].cpu()).abs().max())
            if saved_error>1e-10:raise AssertionError(saved_error)
            rows.append({"source":root.name,"arm":arm,"count":len(points),"lambda":scale,
                         "saved_value_max_error":saved_error,**metrics})
            arrays.update({f"{root.name}/{arm}/{k}":v for k,v in values.items()})
            sources[f"{root.name}/{arm}"]={"path":str(checkpoint),"sha256":digest}
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=digest:raise AssertionError("Source changed")
            print(json.dumps({"source":root.name,"arm":arm,"elapsed_seconds":time.monotonic()-start}),flush=True)
    write_csv(output/"metrics.csv",rows);torch.save(arrays,output/"arrays.pt")
    summary={"sources":sources,"count":count,"scale":scale,"elapsed_seconds":time.monotonic()-start,
             "maximum_trace_partition_error":max(r["trace_partition_max_error"] for r in rows),
             "maximum_saved_value_error":max(r["saved_value_max_error"] for r in rows),
             "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "scope":"Exact local tangent/radial trace partition; frames detached before differentiating network. Remaining ambient trace uses verified saved full response. Not a global-linearity or undertraining claim."}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-roots",nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--count",type=int,default=64)
    print(json.dumps(run(**vars(parser.parse_args())),indent=2),flush=True)


if __name__=="__main__":main()
