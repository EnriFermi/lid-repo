"""Held-out scalar denoising calibration, never fitted to LID labels.

For each fixed scale fit gamma minimizing clean-target prediction MSE:
g(r)=a(r)+gamma*(b(r)-a(r)), with a the identity or training affine projection.
The response changes consistently: tr(Jg)=rank(a)+gamma*(tr(Jb)-rank(a)).
This is a diagnostic of global displacement calibration, not guaranteed to
fix clean-query errors or finite-scale dimension bias.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.response_full_models_20260907_span import sha256
from experiments.response_full_models_20260907_ve import canonical_posterior
from experiments.response_full_models_20260907_vp import load_field
from experiments.response_full_multiscale_20260907 import project
from experiments.response_full_training_20260907 import posterior
from experiments.ve_known_lid_recoverability_campaign import DATASET_SPECS, make_splits
from models.training import load_checkpoint


def fit_coefficient(delta, target_delta):
    denominator=delta.square().sum()
    if denominator<=0: return torch.ones((),dtype=delta.dtype,device=delta.device)
    return (delta*target_delta).sum()/denominator


def calibrated_response(response, gamma, base_rank):
    return base_rank+gamma*(response-base_rank)


def run(args):
    torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=False)
    a=np.load(args.arrays)
    before=sha256(args.checkpoint)
    if args.family=='vp':
        payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        model=load_field(payload,args.kind,args.mode,args.device)
        predict=lambda q,s:posterior(model,q,s,args.kind)
    else:
        result=load_checkpoint(args.checkpoint,device=args.device)
        result.model.double().eval()
        predict=lambda q,s:canonical_posterior(result,q,s)
    _,validation,test=make_splits(args.dataset,train_size=100000,validation_size=1000,test_size=1000,data_seed=0)
    mean=torch.from_numpy(a['normalization_mean']).float()
    rms=float(a['normalization_rms'])
    xval=((torch.from_numpy(validation)-mean)/rms).double().repeat_interleave(args.replicates,dim=0).to(args.device)
    xtest=((torch.from_numpy(test)-mean)/rms).double().to(args.device)
    query=torch.from_numpy(a['queries']).double().to(args.device)
    center=torch.from_numpy(a['span_center']).double().to(args.device)
    basis=torch.from_numpy(a['span_basis']).double().to(args.device)
    generator=torch.Generator(device=args.device).manual_seed(20261412)
    rows,output=[],{}
    dimension=query.shape[1]
    def batch_predict(q,scale):
        return torch.cat([predict(batch,scale) for batch in q.split(256)])
    with torch.no_grad():
        for i,scale in enumerate(a['scales']):
            scale=float(scale)
            rv=xval+scale*torch.randn(xval.shape,dtype=xval.dtype,device=args.device,generator=generator)
            rt=xtest+scale*torch.randn(xtest.shape,dtype=xtest.dtype,device=args.device,generator=generator)
            bv,bt=batch_predict(rv,scale),batch_predict(rt,scale)
            bq=torch.from_numpy(a['posterior'][i]).double().to(args.device)
            for variant in ['original','projected']:
                if variant=='original':
                    av,at,aq=rv,rt,query
                    pv,pt,pq=bv,bt,bq
                    response=torch.from_numpy(a['response'][i]).double().to(args.device)
                    rank=dimension
                else:
                    av,at,aq=[project(v,center,basis) for v in [rv,rt,query]]
                    pv,pt,pq=[project(v,center,basis) for v in [bv,bt,bq]]
                    response=torch.from_numpy(a['response_span'][i]).double().to(args.device)
                    rank=basis.shape[1]
                gamma=fit_coefficient(pv-av,xval-av)
                values=aq+gamma*(pq-aq)
                rnew=calibrated_response(response,gamma,rank)
                cnew=(values-query).square().sum(-1)/scale**2
                cold=(pq-query).square().sum(-1)/scale**2
                gtest=at+gamma*(pt-at)
                d=DATASET_SPECS[args.dataset].true_lid
                row={'dataset':args.dataset,'family':args.family,'kind':args.kind,'mode':args.mode,
                    'variant':variant,'lambda':scale,'gamma':float(gamma),
                    'response_mae':float((response-d).abs().mean()),
                    'full_mae':float((response+cold-d).abs().mean()),
                    'calibrated_response_mae':float((rnew-d).abs().mean()),
                    'calibrated_full_mae':float((rnew+cnew-d).abs().mean()),
                    'test_noisy_risk_original':float((pt-xtest).square().mean()/scale**2),
                    'test_noisy_risk_calibrated':float((gtest-xtest).square().mean()/scale**2)}
                rows.append(row)
                for key,value in [('posterior',values),('response',rnew),('correction',cnew),('full',rnew+cnew),('gamma',gamma)]:
                    output.setdefault(variant+'_'+key,[]).append(value.cpu().numpy())
    assert sha256(args.checkpoint)==before
    np.savez_compressed(args.output/'arrays.npz',query_ids=a['query_ids'],queries=a['queries'],scales=a['scales'],
        **{key:np.stack(value) for key,value in output.items()})
    (args.output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps({'checkpoint_sha256':before,'checkpoint_unchanged':True,
        'source_sha256':sha256(__file__),'arrays_sha256':sha256(args.arrays),'fit':'1000 held-out selection X with independent Gaussian corruption; no LID labels',
        'fit_replicates':args.replicates,'test':'independent noise on1000 testX; clean256 LID reported separately',
        'random_seed':20261412,'rule':'unconstrained least-squares scalar displacement coefficient per fixed scale',
        'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}},indent=2)+'\n')
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--arrays',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset',required=True)
    p.add_argument('--family',choices=['vp','ve'],required=True)
    p.add_argument('--kind',choices=['posterior','noise'],default='noise')
    p.add_argument('--mode',choices=['last','average'],default='average')
    p.add_argument('--replicates',type=int,default=8)
    p.add_argument('--device',default='cuda:1')
    run(p.parse_args())
