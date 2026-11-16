"""Preserved-query exact-law study; no fitting of a posterior to finite samples."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from datasets.archive import EXACT_ARCHIVE_PASSWORD
from experiments.response_full_oracle_20260907 import spiral_posterior,tube_posterior,cylinder_exact,spiral_curve,tube_point
from experiments.ve_known_lid_recoverability_campaign import make_splits


SCALES=np.logspace(-2,0,9)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(arrays,dimension):
    result=[]
    for i,scale in enumerate(arrays['scales']):
        row={'scale':float(scale)}
        for name in ['response','correction','full']:
            x=arrays[name][i]
            row[name+'_mean']=float(x.mean())
            if name!='correction':row[name+'_mae']=float(np.abs(x-dimension).mean())
        row['full_better_fraction']=float((np.abs(arrays['full'][i]-dimension)<np.abs(arrays['response'][i]-dimension)).mean())
        if 'other_turn_mass'in arrays:row['other_turn_mass_mean']=float(arrays['other_turn_mass'][i].mean())
        result.append(row)
    return result


def evaluate(queries,scale,kind,*,metric=None,ambient_map=None,ambient_offset=None,ambient_queries=None,scales=SCALES):
    count=len(queries);active=queries.shape[1]
    arrays={name:np.zeros((len(scales),count)) for name in ['response','correction','full','response_order_error','correction_order_error','response_cutoff_error','correction_cutoff_error']}
    arrays.update(scales=scales,query_ids=np.arange(count),queries_raw=queries,raw_sigma=scales*scale,
                  posterior_raw=np.zeros((len(scales),count,active)))
    if kind=='spiral':
        arrays.update(other_turn_mass=np.zeros((len(scales),count)),local_response=np.zeros((len(scales),count)),local_correction=np.zeros((len(scales),count)),latent_t=1/np.linalg.norm(queries,axis=1))
    else:
        arrays.update(cylinder_response=np.zeros((len(scales),count)),cylinder_correction=np.zeros((len(scales),count)),latent_u=queries[:,0]+4,local_radius=np.linalg.norm(queries[:,1:],axis=1))
    if ambient_map is not None:arrays['posterior']=np.zeros((len(scales),count,ambient_map.shape[1]))
    for i,lam in enumerate(scales):
        sigma=lam*scale
        for j,q in enumerate(queries):
            if kind=='spiral':
                low=spiral_posterior(q,sigma,order=48,metric=metric)
                high=spiral_posterior(q,sigma,order=96,metric=metric)
                tail=spiral_posterior(q,sigma,order=96,cutoff=12,metric=metric)
                local=spiral_posterior(q,sigma,order=96,metric=metric,restrict_turn=True)
                arrays['other_turn_mass'][i,j]=high.metadata['other_turn_mass']
                arrays['local_response'][i,j]=local.response;arrays['local_correction'][i,j]=local.correction
            else:
                low=tube_posterior(q,sigma,order=128)
                high=tube_posterior(q,sigma,order=256)
                tail=tube_posterior(q,sigma,order=256,cutoff=12)
                cylinder=cylinder_exact(q,sigma,np.linalg.norm(q[1:]))
                arrays['cylinder_response'][i,j]=cylinder[1];arrays['cylinder_correction'][i,j]=cylinder[2]
            for name in ['response','correction','full']:arrays[name][i,j]=getattr(high,name)
            for name in ['response','correction']:
                arrays[name+'_order_error'][i,j]=abs(getattr(low,name)-getattr(high,name))
                arrays[name+'_cutoff_error'][i,j]=abs(getattr(tail,name)-getattr(high,name))
            arrays['posterior_raw'][i,j]=high.mean
            if ambient_map is not None:
                posterior=high.mean@ambient_map+ambient_offset
                arrays['posterior'][i,j]=posterior
                arrays['correction'][i,j]=np.square(posterior-ambient_queries[j]).sum()/lam**2
                arrays['full'][i,j]=arrays['response'][i,j]+arrays['correction'][i,j]
        print(kind,float(lam),float(arrays['response'][i].mean()),float(arrays['correction'][i].mean()),flush=True)
    return arrays


def rendered_spiral(output):
    name='e1_spiral_pca'
    root=Path('artifacts/response_full_20260907/rendered_span_first64')/name
    source=torch.load(root/'span_and_queries.pt',weights_only=False,map_location='cpu')
    rawroot=Path('artifacts/spiral_spaghetti_convergence_20260905/run/exact_subset/benchmarks')/name
    coeff={}
    with zipfile.ZipFile('artifacts/spiral_spaghetti_convergence_20260905/benchmarks.zip')as archive:
        for split in ['train','test']:
            member=f'benchmarks/{name}/{split}/coefficients.npy'
            coeff[split]=np.load(io.BytesIO(archive.read(member,pwd=EXACT_ARCHIVE_PASSWORD.encode())),allow_pickle=False)
    np.savez(output/'exact_archive_coefficients.npz',**coeff)
    training=np.load(rawroot/'train/dataset.npy',mmap_mode='r')[:512].reshape(512,-1)
    design=np.column_stack([coeff['train'][:512,:2],np.ones(512)])
    affine=np.linalg.lstsq(design,training,rcond=None)[0]
    matrix,offset=affine[:2],affine[2]
    metric=matrix@matrix.T
    rms=float(source['rms']);mean=source['mean'].numpy().astype(np.float64)
    query=source['queries'].numpy().astype(np.float64)
    image=query*rms+mean
    latent=(image-offset)@matrix.T@np.linalg.inv(metric)
    normal=image-offset-latent@matrix
    test=np.load(rawroot/'test/dataset.npy',mmap_mode='r')[:64].reshape(64,-1)
    facts={'normalization_scale':rms,'gram':metric.tolist(),'train_affine_max_error':float(np.max(np.abs(design@affine-training))),
           'test_affine_max_error':float(np.max(np.abs(coeff['test'][:64,:2]@matrix+offset-test))),
           'normalized_query_reconstruction_max_error':float(np.max(np.abs(image-test))),
           'query_normal_norm_max':float(np.linalg.norm(normal,axis=1).max()),'query_coefficient_max_error':float(np.max(np.abs(latent-coeff['test'][:64,:2])))}
    arrays=evaluate(latent,rms,'spiral',metric=metric,ambient_map=matrix/rms,ambient_offset=(offset-mean)/rms,ambient_queries=query,
                    scales=np.array([.01,.0178,.0316,.0562,.1,.1778,.3162,.5623,1.]))
    arrays.update(queries=query,renderer=matrix,renderer_offset=offset,renderer_gram=metric,query_normal_raw=normal)
    np.savez(output/'rendered_spiral_arrays.npz',**arrays)
    return {'mapping':facts,'summary':summarize(arrays,1),'convergence':{k:float(v.max()) for k,v in arrays.items() if k.endswith('_error')}}


def coefficient_law(output,name):
    active=2 if name=='e1_spiral_pca' else 3
    kind='spiral' if active==2 else 'tube'
    checkpoint=Path('artifacts/ve_known_lid_recoverability/seed-20260904-prospective')/name/'model.pt'
    payload=torch.load(checkpoint,weights_only=False,map_location='cpu')
    rms=float(payload['normalization']['scale']);mean=payload['normalization']['mean'].numpy()
    _,_,test=make_splits(name,train_size=100000,validation_size=1000,test_size=1000,data_seed=0)
    # Preserve the actual float32 normalized model queries, then invert in
    # float64; this includes their tiny departure from the ideal clean curve.
    normalized=((test[:64]-mean)/np.float32(rms)).astype(np.float32)
    raw=normalized.astype(np.float64)*rms+mean.astype(np.float64)
    arrays=evaluate(raw[:,:active],rms,kind)
    arrays.update(queries=normalized,normalization_mean=mean,normalization_scale=np.array(rms))
    posterior=np.zeros((len(SCALES),len(normalized),30))
    posterior[:,:,:active]=arrays['posterior_raw']
    arrays['posterior']=(posterior-mean)/rms
    np.savez(output/(name+'_arrays.npz'),**arrays)
    return {'normalization_scale':rms,'checkpoint_sha256':digest(checkpoint),'summary':summarize(arrays,2 if kind=='tube' else 1),
            'convergence':{k:float(v.max())for k,v in arrays.items()if k.endswith('_error')}}


def fixture_law(output,path):
    if path.suffix=='.npz':
        saved=dict(np.load(path));fixture={'dataset':path.parent.name,'rms':float(saved['normalization_rms']),
              'mean':saved['normalization_mean'],'queries':saved['queries'],'query_ids':saved['query_ids']}
    else:fixture=torch.load(path,weights_only=False,map_location='cpu')
    name=fixture['dataset'];kind='spiral' if name=='e1_spiral_pca' else 'tube';active=2 if kind=='spiral' else 3
    rms=float(fixture['rms']);mean=np.asarray(fixture['mean'],dtype=np.float64)
    normalized=np.asarray(fixture['queries']);raw=normalized.astype(np.float64)*rms+mean
    arrays=evaluate(raw[:,:active],rms,kind)
    posterior=np.zeros((len(SCALES),len(raw),30));posterior[:,:,:active]=arrays['posterior_raw']
    arrays.update(posterior=(posterior-mean)/rms,queries=normalized,query_ids=np.asarray(fixture['query_ids']),
                  normalization_mean=mean,normalization_scale=np.array(rms))
    np.savez(output/'fixture_arrays.npz',**arrays)
    return {'source_fixture':str(path),'fixture_sha256':digest(path),'summary':summarize(arrays,1 if kind=='spiral' else 2),
            'convergence':{k:float(v.max())for k,v in arrays.items()if k.endswith('_error')}}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--mode',choices=['all','rendered','ve','fixture'],default='all');parser.add_argument('--fixture',type=Path);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    report={'query_policy':('exact query_ids and normalization from the supplied fixture' if args.mode=='fixture' else 'fixed test rows0:64, same normalization as each preserved model'),'upstream_revision':'2dcb8e41015f53413ff1ddd049bb006c81a5df52',
            'oracle_source_sha256':digest(Path(__file__).with_name('response_full_oracle_20260907.py')),'study_source_sha256':digest(__file__),'datasets':{}}
    if args.mode in ['all','rendered']:report['datasets']['rendered_spiral']=rendered_spiral(args.output)
    if args.mode in ['all','ve']:
        for name in ['e1_spiral_pca','e6_exp_pca']:report['datasets'][name]=coefficient_law(args.output,name)
    if args.mode=='fixture':report['datasets']['fixture']=fixture_law(args.output,args.fixture)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
