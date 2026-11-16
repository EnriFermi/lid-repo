"""Exact functional source of saved R/C covariance; no optimizer-history inference."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.response_full_exp_profile_20260907 import tube_geometry


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def covariance_terms(posterior, response, query, oracle, scale, blocks=None):
    """Inputs [T,Q,D]/[T,Q]/[Q,D]; covariance centered within each query."""
    b=np.asarray(posterior);r=np.asarray(response)
    mean=b.mean(0); fluct=(b-mean)/scale
    true=(oracle-query)/scale; bias=(mean-oracle)/scale
    rcenter=r-r.mean(0)
    arrays={
        'linear_true_shift':2*np.einsum('qd,tqd->tq',true,fluct),
        'linear_mean_error':2*np.einsum('qd,tqd->tq',bias,fluct),
        'quadratic_motion':(fluct**2).sum(-1),
    }
    c=((b-query)/scale)**2;c=c.sum(-1)
    cbase=(((mean-query)/scale)**2).sum(-1)
    np.testing.assert_allclose(c,cbase+sum(arrays.values()),rtol=1e-11,atol=1e-11)
    summary={}
    for name,v in list(arrays.items()):
        arrays['cov_'+name]=(rcenter*(v-v.mean(0))).mean(0)
        summary['cov_'+name]=float(arrays['cov_'+name].mean())
    cov=(rcenter*(c-c.mean(0))).mean(0)
    covsum=sum(arrays['cov_'+k] for k in ('linear_true_shift','linear_mean_error','quadratic_motion'))
    np.testing.assert_allclose(cov,covsum,rtol=1e-11,atol=1e-11)
    arrays.update(correction=c,covariance=cov,mean_posterior=mean,
                  normalized_fluctuation=fluct,normalized_true_shift=true,normalized_mean_error=bias)
    summary.update(cov_total=float(cov.mean()),identity_max_error=float(np.max(np.abs(cov-covsum))))
    if blocks is not None:
        for name, project in blocks.items():
            pf,pt,pb=project(fluct),project(true),project(bias)
            terms={'linear_true_shift':2*(pf*pt[None]).sum(-1),
                   'linear_mean_error':2*(pf*pb[None]).sum(-1),'quadratic_motion':(pf**2).sum(-1)}
            for term,v in terms.items():
                cc=(rcenter*(v-v.mean(0))).mean(0)
                arrays[f'cov_{term}_{name}']=cc
                summary[f'cov_{term}_{name}']=float(cc.mean())
        for term in ('linear_true_shift','linear_mean_error','quadratic_motion'):
            np.testing.assert_allclose(sum(arrays[f'cov_{term}_{name}'] for name in blocks),
                arrays[f'cov_{term}'],atol=1e-11,rtol=1e-11)
    return summary,arrays


def spatial_blocks(law,oracle):
    if law=='exp':
        frame=tube_geometry(oracle['queries_raw'][:64])['frame']
        blocks={name:(lambda x,v=frame[:,:,i]: np.einsum('...qd,qd->...q',x[...,:3],v)[...,None])
                for i,name in enumerate(('meridional','circumferential','surface_normal'))}
        blocks['ambient']=lambda x:x[...,3:]
        return blocks
    k=np.arange(2,22);phase=oracle['query_phase'][:64]
    tangent=k[None]*np.cos(phase[:,None]*k[None]);tangent/=np.linalg.norm(tangent,axis=-1)[:,None]
    def along(x):return np.einsum('...qd,qd->...q',x[...,:20],tangent)[...,None]
    return {'tangent':along,'curve_normal':lambda x:x[...,:20]-along(x)*tangent,
            'ambient':lambda x:x[...,20:]}


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    paths={'exp':'population_oracle_ve_exp256/fixture_arrays.npz',
           'spaghetti':'population_oracle_ve_spaghetti256/arrays.npz'}
    hashes={};rows=[]
    for law,op in paths.items():
        op=args.base/op;hashes[str(op)]=sha(op);oracle=dict(np.load(op));blocks=spatial_blocks(law,oracle)
        for arm in ('original_lr','low_lr'):
            path=args.base/f've_{law}_refinement/{arm}_temporal.npz';hashes[str(path)]=sha(path);a=dict(np.load(path))
            np.testing.assert_array_equal(a['query_ids'],oracle['query_ids'][:64])
            assert a['response'].shape==(16,4,64)
            indices=[int(np.argmin(np.abs(oracle['scales']-s))) for s in a['scales']]
            np.testing.assert_allclose(a['scales'],oracle['scales'][indices],rtol=1e-9,atol=0)
            saved={}
            for i,(scale,oi) in enumerate(zip(a['scales'],indices)):
                row,arrays=covariance_terms(a['posterior'][:,i],a['response'][:,i],oracle['queries'][:64],
                    oracle['posterior'][oi,:64],float(scale),blocks)
                np.testing.assert_allclose(arrays['correction'],a['correction_original'][:,i],rtol=1e-11,atol=1e-11)
                np.testing.assert_allclose(arrays['mean_posterior'],a['mean_function_posterior'][i],rtol=1e-11,atol=1e-11)
                rows.append({'law':law,'arm':arm,'lambda':float(scale),'queries':64,'snapshots':16,**row})
                for key,v in arrays.items():saved.setdefault(key,[]).append(v)
            np.savez_compressed(args.output/f'{law}_{arm}.npz',query_ids=a['query_ids'],scales=a['scales'],steps=a['steps'],
                **{k:np.stack(v) for k,v in saved.items()})
    (args.output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    with (args.output/'summary.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)
    for path,digest in hashes.items():assert sha(path)==digest
    dependency=Path(__file__).with_name('response_full_exp_profile_20260907.py')
    (args.output/'manifest.json').write_text(json.dumps(dict(source_sha256=sha(__file__),frame_source_sha256=sha(dependency),
        input_hashes=hashes,rows=16,scales='all four stored scales, no selected checkpoint or scale',
        centering='Within each of64 queries over16 saved states, finite population denominator16',
        scale_match='Oracle scale relative mismatch <=1e-9; all displacements use actual model scale denominator',
        provenance='IDs equal oracle first64; every snapshot C and saved mean posterior independently replay',
        interpretation='Exact functional covariance split, not attribution of optimizer training history',
        inputs_unchanged=True,no_model_evaluation_or_training=True),indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=Path('artifacts/response_full_20260907'))
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
