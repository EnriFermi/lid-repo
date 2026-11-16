"""Risk-only calibration of immutable VE refinement states, without fake epochs.

The custom refinement schema is loaded directly; it is never repackaged as a
validation-selected portable checkpoint. The scalar regression is identical
to the earlier held-out calibration, including the shared random-noise tape.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.response_full_calibration_20260907 import fit_coefficient, calibrated_response
from experiments.response_full_models_20260907_span import sha256
from experiments.response_full_models_20260907_ve_refinement import load_refinement_field
from experiments.response_full_multiscale_20260907 import project
from experiments.ve_known_lid_recoverability_campaign import make_splits, DATASET_SPECS
from models.training import diffusion_denoiser_output


def run(args):
    torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=False)
    checkpoint=args.source/'final.pt'
    before=sha256(checkpoint)
    fixture=torch.load(args.fixture,map_location='cpu',weights_only=False)
    _,val,test=make_splits(args.dataset,train_size=100000,validation_size=1000,test_size=1000,data_seed=0)
    mean,rms=fixture['mean'].float(),float(fixture['rms'])
    xv=((torch.from_numpy(val)-mean)/rms).double().repeat_interleave(8,dim=0).to(args.device)
    xt=((torch.from_numpy(test)-mean)/rms).double().to(args.device)
    center,basis=fixture['center'].double().to(args.device),fixture['basis'].double().to(args.device)
    p=basis@basis.T
    rows=[]
    for arm in ['original_lr','low_lr']:
        a=np.load(args.source/f'{arm}_average_arrays.npz')
        ja=np.load(args.audit/f'{arm}_average.npz')
        q=torch.from_numpy(a['queries']).double().to(args.device)
        np.testing.assert_array_equal(a['query_ids'],fixture['query_ids'])
        field=load_refinement_field(checkpoint,arm,mode='average',device=args.device)
        predict=lambda points,scale:diffusion_denoiser_output(field,points,scale,preconditioning='edm',data_std=1.)
        generator=torch.Generator(device=args.device).manual_seed(20261412)
        saved={}
        with torch.no_grad():
            for i,scale in enumerate(a['scales']):
                scale=float(scale)
                rv=xv+scale*torch.randn(xv.shape,dtype=xv.dtype,device=args.device,generator=generator)
                rt=xt+scale*torch.randn(xt.shape,dtype=xt.dtype,device=args.device,generator=generator)
                bv=torch.cat([predict(v,scale) for v in rv.split(256)])
                bt=torch.cat([predict(v,scale) for v in rt.split(256)])
                bq=torch.from_numpy(a['posterior'][i]).double().to(args.device)
                j=torch.from_numpy(ja['jacobian'][i]).double().to(args.device)
                for variant in ['original','projected']:
                    if variant=='original':
                        av,at,aq=rv,rt,q
                        pv,pt,pq=bv,bt,bq
                        response=torch.from_numpy(a['response'][i]).double().to(args.device)
                        base_j=torch.eye(q.shape[-1],dtype=q.dtype,device=args.device)
                        output_j=j
                        rank=q.shape[-1]
                    else:
                        av,at,aq=[project(v,center,basis) for v in [rv,rt,q]]
                        pv,pt,pq=[project(v,center,basis) for v in [bv,bt,bq]]
                        output_j=p@j
                        response=output_j.diagonal(dim1=-2,dim2=-1).sum(-1)
                        base_j=p
                        rank=basis.shape[1]
                    gamma=fit_coefficient(pv-av,xv-av)
                    g=aq+gamma*(pq-aq)
                    rg=calibrated_response(response,gamma,rank)
                    cg=(g-q).square().sum(-1)/scale**2
                    cold=(pq-q).square().sum(-1)/scale**2
                    jg=base_j+gamma*(output_j-base_j)
                    torch.testing.assert_close(jg.diagonal(dim1=-2,dim2=-1).sum(-1),rg,atol=1e-9,rtol=1e-9)
                    gt=at+gamma*(pt-at)
                    d=DATASET_SPECS[args.dataset].true_lid
                    rows.append({'dataset':args.dataset,'arm':arm,'mode':'average','variant':variant,'lambda':scale,
                        'gamma':float(gamma),'response_mae':float((response-d).abs().mean()),
                        'full_mae':float((response+cold-d).abs().mean()),
                        'calibrated_response_mae':float((rg-d).abs().mean()),
                        'calibrated_full_mae':float((rg+cg-d).abs().mean()),
                        'noisy_risk':float((pt-xt).square().mean()/scale**2),
                        'calibrated_noisy_risk':float((gt-xt).square().mean()/scale**2)})
                    for key,value in [('posterior',g),('response',rg),('correction',cg),('full',rg+cg),('jacobian',jg),('gamma',gamma)]:
                        saved.setdefault(variant+'_'+key,[]).append(value.cpu().numpy())
        np.savez_compressed(args.output/f'{arm}_arrays.npz',queries=a['queries'],query_ids=a['query_ids'],scales=a['scales'],
            **{key:np.stack(value) for key,value in saved.items()})
    assert sha256(checkpoint)==before
    (args.output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps({'checkpoint_sha256':before,'checkpoint_unchanged':True,
        'source_sha256':sha256(__file__),'fit':'1000 held-out selectionX times8 independent noise draws; no LID labels',
        'test':'independent1000 noisy testX and original256 clean testqueries',
        'noise_seed':20261412,'model_selection':'fixed last1024 parameter average frompaired4096-step refinements',
        'identity':'derivative is of the calibrated canonical posterior, not unchanged R plus rescaled C'},indent=2)+'\n')
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--dataset',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:1')
    run(p.parse_args())
