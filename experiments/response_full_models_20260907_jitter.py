"""Fixed antithetic posterior smoothing, with correction at the original query.

g(x)=mean_j [b(x+lambda*z_j)+b(x-lambda*z_j)]/2, where tapes are
independent of x and held fixed during spatial differentiation. This changes
the finite-scale posterior on curved laws; matching population-oracle tapes
are mandatory before interpreting it as a useful LID repair.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.response_full_models_20260907_span import projected_trace,sha256
from experiments.response_full_models_20260907_ve import canonical_posterior
from experiments.response_full_models_20260907_vp import load_field
from experiments.response_full_training_20260907 import posterior
from models.training import load_checkpoint


def antithetic_readout(predict,points,scale,tape,batch_size=16):
    if tape.ndim!=3 or tape.shape[0]!=len(points) or tape.shape[2]!=points.shape[1]:
        raise ValueError("Expected tape shaped(query,pair,coordinate)")
    points2=points[:,None,None,:]+scale*torch.stack([tape,-tape],dim=2)
    shape=points2.shape
    value,trace=projected_trace(predict,points2.reshape(-1,shape[-1]),
        torch.eye(shape[-1],device=points.device,dtype=points.dtype),batch_size)
    value=value.reshape(shape);trace=trace.reshape(shape[:-1])
    g=value.mean((1,2));response=trace.mean((1,2))
    correction=(g-points).square().sum(-1)/scale**2
    centered_variation=(value-g[:,None,None,:]).square().sum(-1).mean((1,2))/scale**2
    return {"posterior":g,"response":response,"correction_original":correction,
        "full":response+correction,"individual_posteriors":value,
        "individual_responses":trace,"posterior_variation_over_lambda_squared":centered_variation}


def run(args):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True,exist_ok=False)
    source=np.load(args.model_arrays)
    queries=source["queries"][:args.count]
    ids=source["query_ids"][:args.count]
    tape=np.random.default_rng(args.seed).standard_normal((len(queries),args.pairs,queries.shape[-1]))
    np.savez_compressed(args.output/"noise_tape.npz",queries=queries,query_ids=ids,
        noise=tape,scales=np.asarray(args.scales),query_raw=source["query_raw"][:args.count],
        normalization_mean=source["normalization_mean"] if "normalization_mean"in source else np.empty(0),
        normalization_rms=source["normalization_rms"] if "normalization_rms"in source else np.asarray(np.nan))
    print(json.dumps({"noise_tape_ready":str(args.output/"noise_tape.npz"),"count":len(queries),"pairs":args.pairs}),flush=True)
    if args.family=="ve":
        result=load_checkpoint(args.checkpoint,device=args.device);result.model.double().eval()
        predict=lambda q,s:canonical_posterior(result,q,s)
    else:
        payload=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
        kind=args.family.removeprefix("vp_")
        field=load_field(payload,kind,"average",args.device)
        predict=lambda q,s:posterior(field,q,s,kind)
    before=sha256(args.checkpoint)
    points=torch.from_numpy(queries).double().to(args.device)
    noise=torch.from_numpy(tape).double().to(args.device)
    arrays={"queries":queries,"query_ids":ids,"scales":np.asarray(args.scales),"query_raw":source["query_raw"][:args.count]}
    accumulated={};rows=[]
    for scale in args.scales:
        candidates=np.flatnonzero(np.isclose(source["scales"],scale,rtol=1e-12,atol=0))
        if len(candidates)!=1:raise ValueError("Must choose a stored baseline scale")
        index=int(candidates[0])
        data=antithetic_readout(lambda q:predict(q,scale),points,scale,noise,args.batch_size)
        for key,value in data.items():accumulated.setdefault(key,[]).append(value.cpu().numpy())
        baseline_b=source["posterior"][index,:len(points)]
        baseline_r=source["response"][index,:len(points)]
        baseline_c=source["correction_original"][index,:len(points)]
        baseline_full=baseline_r+baseline_c
        row={"family":args.family,"lambda":scale,"count":len(points),"pairs":args.pairs,
            "baseline_response_d_mae":float(np.abs(baseline_r-args.true_lid).mean()),
            "baseline_full_d_mae":float(np.abs(baseline_full-args.true_lid).mean()),
            "jitter_response_d_mae":float((data["response"]-args.true_lid).abs().mean()),
            "jitter_full_d_mae":float((data["full"]-args.true_lid).abs().mean()),
            "baseline_correction_mean":float(baseline_c.mean()),
            "jitter_correction_mean":float(data["correction_original"].mean()),
            "mean_output_shift_norm_over_lambda":float((data["posterior"]-torch.from_numpy(baseline_b).to(points)).norm(dim=-1).mean()/scale),
            "posterior_variation_over_lambda_squared":float(data["posterior_variation_over_lambda_squared"].mean())}
        rows.append(row);print(json.dumps(row),flush=True)
    arrays.update({key:np.stack(value) for key,value in accumulated.items()})
    np.savez_compressed(args.output/"arrays.npz",**arrays)
    (args.output/"summary.json").write_text(json.dumps(rows,indent=2)+"\n")
    assert sha256(args.checkpoint)==before
    (args.output/"manifest.json").write_text(json.dumps({"source_sha256":sha256(__file__),
        "checkpoint":str(args.checkpoint),"checkpoint_sha256":before,"family":args.family,
        "model_arrays_sha256":sha256(args.model_arrays),"noise_tape_sha256":sha256(args.output/"noise_tape.npz"),
        "seed":args.seed,"rng":"NumPy default_rng private generator independent of training draw streams",
        "pairs":args.pairs,"amplitude":1.,"queries":"First64 of the previously fixed shared256 IDs",
        "definition":"Average b at x±lambda*z; response averages exact derivatives holding tape fixed; correction uses averageb MINUS original x",
        "not_used":"No averaging of full readouts anchored at jittered points",
        "caveat":"Changes exact finite-scale posterior on curved distributions; needs same-tape oracle comparison",
        "all_checkpoints_unchanged":True},indent=2)+"\n")
    return {"output":str(args.output),"rows":len(rows)}


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-arrays",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--family",choices=["ve","vp_noise","vp_posterior"],required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--true-lid",type=float,default=1.)
    p.add_argument("--scales",nargs="+",type=float,default=[.01,.1,.316227766,1.])
    p.add_argument("--count",type=int,default=64)
    p.add_argument("--pairs",type=int,default=8)
    p.add_argument("--seed",type=int,default=20260907)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--batch-size",type=int,default=16)
    return p


if __name__=="__main__":print(json.dumps(run(parser().parse_args())),flush=True)
