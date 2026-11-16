"""Local tangent/normal decomposition on the exact one-parameter Spaghetti law.

The global affine span has rank20, whereas the tangent is one-dimensional.
Geometric directions are oracle diagnostics only, not a proposed estimator.
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


def tangent_from_phase(phase):
    frequencies=np.arange(2,22,dtype=np.float64)
    velocity=np.cos(np.asarray(phase)[:,None]*frequencies)*frequencies
    return velocity/np.linalg.norm(velocity,axis=-1,keepdims=True)


def split_jacobian_error(error,tangent):
    v=np.asarray(tangent)
    tt=np.einsum("bi,bij,bj->b",v,error,v)
    ev=np.einsum("bij,bj->bi",error,v)
    ve=np.einsum("bi,bij->bj",v,error)
    nt=ev-tt[:,None]*v
    tn=ve-tt[:,None]*v
    nn=error-tt[:,None,None]*v[:,:,None]*v[:,None,:]-nt[:,:,None]*v[:,None,:]-v[:,:,None]*tn[:,None,:]
    terms={"tangent_tangent_jacobian_error_squared":tt**2,
        "normal_output_tangent_input_jacobian_error_squared":(nt**2).sum(-1),
        "tangent_output_normal_input_jacobian_error_squared":(tn**2).sum(-1),
        "normal_normal_jacobian_error_squared":(nn**2).sum((-1,-2))}
    np.testing.assert_allclose(sum(terms.values()),(error**2).sum((-1,-2)),rtol=1e-11,atol=1e-11)
    return terms


def active_jacobian(predict,queries,active=20,batch_size=16):
    values,blocks=[],[]
    basis=torch.eye(queries.shape[1],device=queries.device,dtype=queries.dtype)[:active]
    for batch in queries.split(batch_size):
        x=batch.detach().requires_grad_(True)
        output=predict(x)
        probes=basis[:,None,:].expand(-1,len(x),-1)
        grad=torch.autograd.grad(output,x,grad_outputs=probes,is_grads_batched=True)[0]
        values.append(output.detach());blocks.append(grad.permute(1,0,2).detach())
    return torch.cat(values),torch.cat(blocks)


def run(args):
    torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=False)
    model_arrays=np.load(args.model_arrays);oracle=np.load(args.oracle_arrays)
    np.testing.assert_array_equal(model_arrays["queries"],oracle["queries"])
    np.testing.assert_array_equal(model_arrays["query_ids"],oracle["query_ids"])
    np.testing.assert_allclose(model_arrays["scales"],oracle["scales"],rtol=1e-9,atol=0)
    queries=torch.from_numpy(model_arrays["queries"]).double().to(args.device)
    tangent=tangent_from_phase(oracle["query_phase"])
    if args.family=="ve":
        result=load_checkpoint(args.checkpoint,device=args.device);result.model.double().eval()
        predict=lambda q,s:canonical_posterior(result,q,s)
    else:
        kind=args.family.removeprefix("vp_")
        payload=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
        field=load_field(payload,kind,args.mode,args.device)
        predict=lambda q,s:posterior(field,q,s,kind)
    before=sha256(args.checkpoint)
    rows,accumulated=[],{}
    for i,scale in enumerate(model_arrays["scales"]):
        value,jac=active_jacobian(lambda q:predict(q,float(scale)),queries)
        value,jac=value.cpu().numpy(),jac.cpu().numpy()
        np.testing.assert_allclose(value,model_arrays["posterior"][i],rtol=1e-9,atol=1e-9)
        learned=jac[:,:,:20];truth=oracle["jacobian_active"][i]
        response=np.trace(learned,axis1=-2,axis2=-1)
        np.testing.assert_allclose(response,model_arrays["response_span"][i],rtol=1e-9,atol=1e-9)
        tangent_r=np.einsum("bi,bij,bj->b",tangent,learned,tangent)
        oracle_tangent_r=np.einsum("bi,bij,bj->b",tangent,truth,tangent)
        oracle_r=np.trace(truth,axis1=-2,axis2=-1)
        error=value-oracle["posterior"][i]
        tangent_e=(error[:,:20]*tangent).sum(-1)
        tangent_energy=tangent_e**2/scale**2
        span_energy=(error[:,:20]**2).sum(-1)/scale**2
        normal_energy=span_energy-tangent_energy
        ambient_energy=(error[:,20:]**2).sum(-1)/scale**2
        jerror=learned-truth
        symmetric=.5*(learned+learned.transpose(0,2,1))
        eigen=np.linalg.eigvalsh(symmetric)
        data={"learned_jacobian_active":learned,"jacobian_error_active":jerror,
            "response_tangent":tangent_r,"response_curve_normal":response-tangent_r,
            "oracle_response_tangent":oracle_tangent_r,"oracle_response_curve_normal":oracle_r-oracle_tangent_r,
            "posterior_error_tangent_energy":tangent_energy,
            "posterior_error_curve_normal_energy":normal_energy,
            "posterior_error_ambient_normal_energy":ambient_energy,
            "jacobian_symmetric_eigenvalues":eigen,
            "jacobian_error_singular_values":np.linalg.svd(jerror,compute_uv=False),
            "jacobian_antisymmetric_frobenius_squared":((.5*(learned-learned.transpose(0,2,1)))**2).sum((-1,-2)),
            **split_jacobian_error(jerror,tangent)}
        for key,value in data.items():accumulated.setdefault(key,[]).append(value)
        rows.append({"lambda":float(scale),"count":len(queries),
            "response_tangent_mean":float(tangent_r.mean()),
            "response_curve_normal_mean":float((response-tangent_r).mean()),
            "oracle_response_tangent_mean":float(oracle_tangent_r.mean()),
            "oracle_response_curve_normal_mean":float((oracle_r-oracle_tangent_r).mean()),
            "response_tangent_oracle_mae":float(np.abs(tangent_r-oracle_tangent_r).mean()),
            "response_curve_normal_oracle_mae":float(np.abs(response-tangent_r-oracle_r+oracle_tangent_r).mean()),
            "mean_negative_symmetric_eigenvalue_count":float((eigen<0).sum(-1).mean()),
            **{key+"_mean":float(value.mean()) for key,value in data.items() if value.ndim==1}})
        print(json.dumps(rows[-1]),flush=True)
    np.savez_compressed(args.output/"arrays.npz",query_ids=oracle["query_ids"],scales=model_arrays["scales"],
        query_phase=oracle["query_phase"],tangent=tangent,**{k:np.stack(v) for k,v in accumulated.items()})
    (args.output/"summary.json").write_text(json.dumps(rows,indent=2)+"\n")
    with (args.output/"summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    assert sha256(args.checkpoint)==before
    (args.output/"manifest.json").write_text(json.dumps({"source_sha256":sha256(__file__),
        "checkpoint":str(args.checkpoint),"checkpoint_sha256":before,
        "model_arrays_sha256":sha256(args.model_arrays),"oracle_arrays_sha256":sha256(args.oracle_arrays),
        "active_coordinates":"Exact generator first20 coefficients; remaining10 identicallyzero",
        "geometry":"1D tangent from exact latentphase;19 curve-normal directions inside20Dspan;10 ambient-null directions",
        "family":args.family,"mode":args.mode,
        "intervention":"None: oracle geometric diagnostic only","dtype":"float64 exact batchedVJPs",
        "all_checkpoints_unchanged":True},indent=2)+"\n")
    return {"rows":len(rows),"output":str(args.output)}


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-arrays",type=Path,default=Path("artifacts/response_full_20260907/ve_spaghetti_shared256/e8_spaghetti_pca/arrays.npz"))
    p.add_argument("--oracle-arrays",type=Path,default=Path("artifacts/response_full_20260907/population_oracle_ve_spaghetti256/arrays.npz"))
    p.add_argument("--checkpoint",type=Path,default=Path("artifacts/ve_known_lid_recoverability/seed-20260904-prospective/e8_spaghetti_pca/model.pt"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--family",choices=["ve","vp_noise","vp_posterior"],default="ve")
    p.add_argument("--mode",choices=["last","average"],default="average")
    return p


if __name__=="__main__":print(json.dumps(run(parser().parse_args())),flush=True)
