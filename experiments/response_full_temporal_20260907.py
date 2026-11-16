"""Exact correction decomposition across saved finite-training-window values.

These arrays contain no snapshot derivatives. This script NEVER invents the
response/full of their function average. Last/weight-average full scores are
separate, previously evaluated checkpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.response_full_models_20260907_span import sha256


def temporal_correction(values, queries, scales):
    values,queries,scales=values.double(),queries.double(),scales.double()
    mean=values.mean(0)
    per_snapshot=(values-queries).square().sum(-1)/scales[None,:,None]**2
    at_mean=(mean-queries).square().sum(-1)/scales[:,None]**2
    variance=(values-mean).square().sum(-1).mean(0)/scales[:,None]**2
    torch.testing.assert_close(per_snapshot.mean(0),at_mean+variance,rtol=1e-10,atol=1e-10)
    return {'snapshot_correction':per_snapshot,'mean_prediction':mean,
        'correction_of_mean_prediction':at_mean,'correction_jensen_excess':variance}


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    temporal=torch.load(args.source/'temporal.pt',map_location='cpu',weights_only=False)
    fixture=torch.load(args.source/'fixture.pt',map_location='cpu',weights_only=False)
    rows,arrays=[],{}
    metrics=json.loads((args.source/'metrics.json').read_text())
    for kind,values in temporal['predictions'].items():
        out=temporal_correction(values,fixture['queries'],torch.as_tensor(temporal['scales']))
        arrays.update({kind+'_'+key:value.numpy() for key,value in out.items()})
        for i,scale in enumerate(temporal['scales']):
            mean_c=out['snapshot_correction'][:,i].mean()
            excess=out['correction_jensen_excess'][i].mean()
            row={'dataset':fixture['dataset'],'kind':kind,'lambda':float(scale),
                'snapshot_count':len(values),'snapshot_mean_correction':float(mean_c),
                'correction_of_mean_prediction':float(out['correction_of_mean_prediction'][i].mean()),
                'jensen_excess':float(excess),'jensen_fraction':float(excess/mean_c)}
            for mode in ['last','average']:
                item=next(m for m in metrics if m['kind']==kind and m['mode']==mode and abs(m['lambda']-float(scale))<1e-8)
                for key in ['response_mae','full_mae','correction_mean']:row[mode+'_'+key]=item[key]
            rows.append(row)
    np.savez_compressed(args.output/'arrays.npz',query_ids=fixture['query_ids'],steps=temporal['steps'],
        scales=np.asarray(temporal['scales']),**arrays)
    (args.output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps({'source_sha256':sha256(__file__),
        'temporal_sha256':sha256(args.source/'temporal.pt'),'fixture_sha256':sha256(args.source/'fixture.pt'),
        'metrics_sha256':sha256(args.source/'metrics.json'),
        'scope':'checkpoint-window variation, not training-seed uncertainty or independent-error variance',
        'missing':'no per-snapshot derivatives; mean-function response/full not available',
        'mean_function_and_parameter_average':'distinct objects; only correction of each is directly compared'},indent=2)+'\n')
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
