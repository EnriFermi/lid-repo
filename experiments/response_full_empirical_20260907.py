"""Exact posterior of the actual finite training distribution plus Gaussian noise.

This is the Bayes optimum of empirical denoising risk, not a trained network
and not the continuous population oracle. Nested sample counts diagnose finite
data limitations separately from neural approximation and finite noise scale.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.response_full_models_20260907_span import sha256
from experiments.ve_known_lid_recoverability_campaign import DATASET_SPECS, make_splits


def empirical_posterior(train, queries, scale, batch_size=16):
    """Float64 centered moments avoid subtraction of large uncentered moments."""
    result = {key: [] for key in ['posterior', 'response', 'correction', 'full', 'effective_points']}
    with torch.no_grad():
        for query in queries.split(batch_size):
            delta = train[None] - query[:,None]
            distance = delta.square().sum(-1)
            weights = torch.softmax(-distance/(2*scale**2), dim=1)
            displacement = torch.einsum('bn,bnd->bd', weights, delta)
            full = (weights*distance).sum(-1)/scale**2
            correction = displacement.square().sum(-1)/scale**2
            values = {'posterior':query+displacement, 'response':full-correction,
                'correction':correction, 'full':full, 'effective_points':1/weights.square().sum(-1)}
            for key,value in values.items(): result[key].append(value.cpu())
    return {key:torch.cat(value) for key,value in result.items()}


def run(args):
    torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=False)
    fixture = torch.load(args.fixture,map_location='cpu',weights_only=False)
    dataset = fixture['dataset']
    train,_,test = make_splits(dataset,train_size=100000,validation_size=1000,test_size=1000,data_seed=0)
    ids = np.asarray(fixture['query_ids'])[:args.query_count]
    np.testing.assert_array_equal(test[ids],np.asarray(fixture['query_raw'])[:len(ids)])
    q = fixture['queries'][:len(ids)].double()
    # Use every varying coefficient. Global null coordinates are exactly absent
    # and do not affect kernel weights, response, or correction.
    active = np.flatnonzero(np.ptp(train,axis=0)>0)
    x = ((torch.from_numpy(train)-fixture['mean'])/fixture['rms']).double()
    x = x[:,active].to(args.device)
    queries = q[:,active].to(args.device)
    scales = np.asarray([.01,.0177827941,.0316227766,.0562341325,.1,.177827941,.316227766,.562341325,1.])
    rows=[]
    d=DATASET_SPECS[dataset].true_lid
    for count in args.counts:
        arrays={}
        for scale in scales:
            out=empirical_posterior(x[:count],queries,float(scale),args.batch_size)
            for key,value in out.items(): arrays.setdefault(key,[]).append(value.numpy())
            rows.append({'dataset':dataset,'training_count':count,'query_count':len(ids),'lambda':float(scale),
                'response_mae':float((out['response']-d).abs().mean()),
                'full_mae':float((out['full']-d).abs().mean()),
                **{key+'_mean':float(out[key].mean()) for key in ['response','correction','effective_points']},
                'effective_points_median':float(out['effective_points'].median()),
                'fraction_effective_points_lt_10':float((out['effective_points']<10).double().mean())})
        np.savez_compressed(args.output/f'n{count}_arrays.npz',query_ids=ids,query_raw=test[ids],queries=q.numpy(),
            active_coordinates=active,scales=scales,**{key:np.stack(value) for key,value in arrays.items()})
        print(json.dumps(rows[-len(scales):]),flush=True)
    (args.output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps({'fixture_sha256':sha256(args.fixture),
        'source_sha256':sha256(__file__),'device':args.device,'dtype':'float64',
        'training_counts':args.counts,'selection':'fixed nested training prefixes; no LID tuning',
        'meaning':'Bayes optimum for uniform finite training distribution, not continuous population',
        'normalization':'replays original float32 canonical train/query normalization before float64 moments'},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--counts',nargs='+',type=int,default=[10000,100000])
    p.add_argument('--query-count',type=int,default=256)
    p.add_argument('--batch-size',type=int,default=8)
    run(p.parse_args())
