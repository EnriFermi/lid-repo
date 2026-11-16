"""Does an ambient-noise shell expose clean-query normal-derivative error?

The perturbations separate the GLOBAL affine data span and its orthogonal
complement, not a local tangent of a curved manifold. The exact posterior is
independent of every complementary coordinate, at all normal radii. Thus its
normal value and normal Jacobian block are known without local-LID labels or
population quadrature. This is a fixed-model forward/derivative diagnostic.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from experiments.response_full_models_20260907_span import sha256
from experiments.response_full_models_20260907_ve import canonical_posterior
from experiments.response_full_models_20260907_vp import load_field
from experiments.response_full_training_20260907 import posterior
from models.training import load_checkpoint


def normal_blocks(predict, points, normal_basis, center, scale, batch_size=16):
    """Exact N^T J_b N block and normal posterior error, query by query."""
    values, blocks = [], []
    for batch in points.split(batch_size):
        x = batch.detach().requires_grad_(True)
        output = predict(x)
        probes = normal_basis.T[:, None, :].expand(-1, len(x), -1)
        gradients = torch.autograd.grad(output, x, grad_outputs=probes, is_grads_batched=True)[0]
        block = torch.einsum("kbd,dl->bkl", gradients, normal_basis)
        values.append(((output-center)@normal_basis/scale).detach())
        blocks.append(block.detach())
    value, block = torch.cat(values), torch.cat(blocks)
    diagonal = block.diagonal(dim1=-2, dim2=-1)
    trace = diagonal.sum(-1)
    return {"normal_value_over_lambda": value, "normal_jacobian": block,
        "normal_trace": trace, "normal_trace_per_dimension": diagonal.mean(-1),
        "normal_abs_diagonal_per_dimension": diagonal.abs().mean(-1),
        "positive_diagonal_fraction": (diagonal>0).double().mean(-1),
        "trace_over_absolute_diagonal_sum": trace.abs()/diagonal.abs().sum(-1).clamp_min(1e-30),
        "normal_jacobian_frobenius_squared_per_dimension": block.square().sum((-1,-2))/block.shape[-1],
        "normal_value_energy_over_lambda_squared": value.square().sum(-1)}


def run(args):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=False)
    records, rows = [], []
    for law in args.laws:
        source = args.root/f"vp_{law}_paired"
        span_path = args.root/f"vp_{law}_span/span_and_queries.pt"
        fixture = torch.load(span_path, map_location="cpu", weights_only=False)
        queries, center, basis = fixture["queries"].double(), fixture["center"].double(), fixture["basis"].double()
        dimension, rank = queries.shape[1], basis.shape[1]
        normal = torch.linalg.qr(basis, mode="complete").Q[:, rank:]
        rng = torch.Generator().manual_seed(args.noise_seed)
        noise = torch.randn(queries.shape, generator=rng, dtype=torch.float64)
        span_noise = (noise@basis)@basis.T
        normal_noise = (noise@normal)@normal.T
        queries = center+((queries-center)@basis)@basis.T
        torch.save({"query_ids": fixture["query_ids"], "queries": queries,
            "normal_noise": normal_noise, "span_noise": span_noise,
            "basis": basis, "normal_basis": normal, "center": center}, args.output/f"{law}_probe_fixture.pt")
        checkpoint = source/"final.pt"
        before = sha256(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        models = []
        for kind in args.kinds:
            field = load_field(payload, kind, "average", args.device)
            models.append((f"vp_{kind}", lambda q, s, field=field, kind=kind: posterior(field,q,s,kind),
                queries, center, checkpoint, before))
        if args.include_ve:
            dataset = payload["config"]["dataset"]
            path = args.ve_source/dataset/"model.pt"
            result = load_checkpoint(path, device=args.device)
            result.model.double().eval()
            # Same raw query/noise coordinate directions, respective physical
            # normalized scales and normalization contract remain explicit.
            ve_queries = ((torch.as_tensor(fixture["query_raw"])-result.normalization_mean.cpu())/result.normalization_scale).double()
            ve_center = (fixture["mean"].double()+float(fixture["rms"])*center-result.normalization_mean.cpu().double())/result.normalization_scale
            ve_queries = ve_center+((ve_queries-ve_center)@basis)@basis.T
            models.append(("ve_edm", lambda q,s,result=result:canonical_posterior(result,q,s),
                ve_queries,ve_center,path,sha256(path)))
        for name, predict, model_queries, model_center, model_path, digest in models:
            accumulated = {}
            for span_factor in args.span_factors:
                for scale in args.scales:
                    for rho in args.rhos:
                        points = model_queries+scale*(span_factor*span_noise+rho*normal_noise)
                        data = normal_blocks(lambda q:predict(q,scale), points.to(args.device),
                            normal.to(args.device),model_center.to(args.device),scale,args.batch_size)
                        row = {"law":law,"model":name,"lambda":scale,
                            "global_span_noise_factor":span_factor,"normal_noise_factor":rho,
                            "count":len(queries),"normal_dimension":dimension-rank,
                            "normal_input_rms_norm_over_lambda":float((rho*normal_noise).square().sum(-1).mean().sqrt()),
                            "normal_value_rms_norm_over_lambda":float(data["normal_value_energy_over_lambda_squared"].mean().sqrt()),
                            "normal_trace_mean":float(data["normal_trace"].mean()),
                            "normal_trace_mae":float(data["normal_trace"].abs().mean()),
                            "normal_trace_p10":float(data["normal_trace"].quantile(.1)),
                            "normal_trace_p90":float(data["normal_trace"].quantile(.9)),
                            **{key+"_mean":float(value.mean()) for key,value in data.items() if value.ndim==1}}
                        rows.append(row)
                        for key,value in data.items():
                            accumulated.setdefault(key,[]).append(value.cpu().numpy())
                        print(json.dumps(row),flush=True)
            np.savez_compressed(args.output/f"{law}_{name}_arrays.npz",
                query_ids=fixture["query_ids"],scales=np.asarray(args.scales),rhos=np.asarray(args.rhos),
                span_factors=np.asarray(args.span_factors),
                **{key:np.stack(value).reshape(len(args.span_factors),len(args.scales),len(args.rhos),*value[0].shape)
                   for key,value in accumulated.items()})
            assert sha256(model_path)==digest
            records.append({"law":law,"model":name,"checkpoint":str(model_path),
                "checkpoint_sha256":digest,"probe_basis_source_sha256":sha256(span_path),
                "normal_dimension":dimension-rank})
        assert sha256(checkpoint)==before
    (args.output/"summary.json").write_text(json.dumps(rows,indent=2)+"\n")
    with (args.output/"summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (args.output/"manifest.json").write_text(json.dumps({
        "source_sha256":sha256(__file__),"noise_seed":args.noise_seed,
        "noise_contract":"Paired standard Gaussian draw projected into global affine span and complement; no per-query unit-norm rescaling",
        "rhos":args.rhos,"scales":args.scales,"global_span_noise_factors":args.span_factors,
        "normal_oracle_contract":"N^T(b_star-center)=0 and N^T J_b_star N=0 for every ambient normal coordinate",
        "normal_basis":"QR completion of training-only affine-span SVD; does not identify local tangent",
        "dtype":"float64 forward and exact normal-block VJPs","device":args.device,
        "all_checkpoints_unchanged":True,"models":records},indent=2)+"\n")
    return {"rows":len(rows),"output":str(args.output)}


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("artifacts/response_full_20260907"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--laws",nargs="+",default=["spiral","exp"])
    p.add_argument("--kinds",nargs="+",default=["posterior","noise"])
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--scales",nargs="+",type=float,default=[.01,.1,.316227766,1.])
    p.add_argument("--rhos",nargs="+",type=float,default=[0.,.25,.5,1.,1.5])
    p.add_argument("--span-factors",nargs="+",type=float,default=[0.,1.])
    p.add_argument("--noise-seed",type=int,default=20261009)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--include-ve",action="store_true")
    p.add_argument("--ve-source",type=Path,default=Path("artifacts/ve_known_lid_recoverability/seed-20260904-prospective"))
    return p


if __name__=="__main__":print(json.dumps(run(parser().parse_args())),flush=True)
