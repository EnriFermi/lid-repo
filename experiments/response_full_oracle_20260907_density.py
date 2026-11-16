"""Same Exp surface and queries; only population sampling density changes."""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from experiments.response_full_oracle_20260907 import tube_point,tube_posterior


def local_coordinates(query,mean,sigma):
    radius=np.linalg.norm(query[1:]);direction=query[1:]/radius
    tangent=np.concatenate(([1.],-radius*direction))/np.sqrt(1+radius**2)
    normal=np.concatenate(([radius],direction))/np.sqrt(1+radius**2)
    delta=(mean-query)/sigma
    return float(delta@tangent),float(delta@normal)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    source=dict(np.load(a.input));q=source['queries_raw'];sigmas=source['raw_sigma'];shape=(len(sigmas),len(q))
    arrays={key:np.zeros(shape)for key in ['uniform_response','uniform_correction','area_response','area_correction','uniform_tangent_shift_over_sigma','uniform_normal_shift_over_sigma','area_tangent_shift_over_sigma','area_normal_shift_over_sigma','order_error']}
    arrays['uniform_posterior_raw']=np.zeros(shape+(3,));arrays['area_posterior_raw']=np.zeros(shape+(3,))
    for i,sigma in enumerate(sigmas):
        for j,query in enumerate(q):
            old=tube_posterior(query,sigma,order=256);new=tube_posterior(query,sigma,order=256,prior='uniform_area');low=tube_posterior(query,sigma,order=128,prior='uniform_area')
            for label,result in [('uniform',old),('area',new)]:
                arrays[label+'_response'][i,j]=result.response;arrays[label+'_correction'][i,j]=result.correction;arrays[label+'_posterior_raw'][i,j]=result.mean
                tangent,normal=local_coordinates(query,result.mean,sigma)
                arrays[label+'_tangent_shift_over_sigma'][i,j]=tangent;arrays[label+'_normal_shift_over_sigma'][i,j]=normal
            arrays['order_error'][i,j]=max(abs(new.response-low.response),abs(new.correction-low.correction))
    arrays.update(query_ids=source['query_ids'],queries_raw=q,scales=source['scales'],raw_sigma=sigmas)
    np.savez(a.output/'arrays.npz',**arrays)
    summary=[]
    for i,scale in enumerate(source['scales']):
        row={'scale':float(scale)}
        for label in ['uniform','area']:
            response=arrays[label+'_response'][i];correction=arrays[label+'_correction'][i]
            row[label+'_response_mae']=float(np.abs(response-2).mean());row[label+'_full_mae']=float(np.abs(response+correction-2).mean())
            row[label+'_tangent_c_mean']=float(np.square(arrays[label+'_tangent_shift_over_sigma'][i]).mean());row[label+'_normal_c_mean']=float(np.square(arrays[label+'_normal_shift_over_sigma'][i]).mean())
        summary.append(row)
    controls=[]
    for u in [0.,2.,4.,6.,8.]:
        for sigma in [.00442778,.0442778,.14]:
            query=tube_point(u,.7);row={'u':u,'raw_sigma':sigma}
            for label,prior in [('uniform','uniform_u'),('area','uniform_area')]:
                r=tube_posterior(query,sigma,order=256,prior=prior);t,n=local_coordinates(query,r.mean,sigma)
                row[label]={'response':r.response,'correction':r.correction,'full':r.full,'tangent_shift_over_sigma':t,'normal_shift_over_sigma':n,'axial_mean_shift':float(r.mean[0]-query[0])}
            controls.append(row)
    report={'source':str(a.input),'source_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),'intervention':'same X(u,theta) surface; prior density in u changes from constant to a(u)*sqrt(1+a(u)^2); theta remains uniform',
            'evaluation_distribution':'unchanged original uniform-u held-out256queries; not newly sampled uniform-area queries','max_area_128_to_256_readout_error':float(arrays['order_error'].max()),
            'max_original_response_replay_error':float(np.abs(arrays['uniform_response']-source['response']).max()),'summary':summary,'controls':controls}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
