"""Fixed J-eigenvalue>=1/4 correction gate, tested without threshold tuning."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from experiments.response_full_oracle_20260907 import spiral_posterior,tube_posterior,tube_point
from experiments.response_full_oracle_20260907_crescent import crescent_posterior


def gated_correction(displacement,sigma,covariance,threshold=.25):
    jacobian=np.asarray(covariance)/sigma**2
    eigenvalues,eigenvectors=np.linalg.eigh(jacobian)
    components=np.asarray(displacement)@eigenvectors/sigma
    kept=eigenvalues>=threshold
    return float(np.square(components[kept]).sum()),eigenvalues,components,eigenvectors


def evaluate(path,output,kind):
    source=dict(np.load(path));queries=source['queries_raw'];sigmas=source['raw_sigma'];shape=(len(sigmas),len(queries));active=queries.shape[1]
    response=np.zeros(shape);correction=np.zeros(shape);gated=np.zeros(shape);cov=np.zeros(shape+(active,active));jacs=np.zeros_like(cov);eig=np.zeros(shape+(active,));components=np.zeros_like(eig)
    oracle={'spiral':spiral_posterior,'tube':tube_posterior,'crescent':crescent_posterior}[kind]
    for i,sigma in enumerate(sigmas):
        for j,q in enumerate(queries):
            result=oracle(q,sigma) if kind=='crescent' else oracle(q,sigma,order=192 if kind=='spiral' else 256)
            response[i,j]=result.response;correction[i,j]=result.correction;cov[i,j]=result.posterior_covariance
            gated[i,j],eig[i,j],components[i,j],_=gated_correction(result.mean-q,sigma,result.posterior_covariance)
            jacs[i,j]=result.posterior_covariance/sigma**2
    d={'spiral':1,'tube':2,'crescent':3}[kind]
    np.savez(output/(kind+'_arrays.npz'),query_ids=source['query_ids'],queries_raw=queries,scales=source['scales'],raw_sigma=sigmas,
             response=response,correction=correction,gated_correction=gated,full=response+correction,gated_full=response+gated,
             covariance_raw=cov,jacobian_active=jacs,eigenvalues=eig,displacement_eigencomponents_over_sigma=components)
    return {'source':str(path),'threshold':.25,'true_local_dimension':d,'max_original_response_replay_error':float(np.abs(response-source['response']).max()),
            'summary':[{'scale':float(s),'response_mae':float(np.abs(response[i]-d).mean()),'full_mae':float(np.abs(response[i]+correction[i]-d).mean()),
                         'gated_full_mae':float(np.abs(response[i]+gated[i]-d).mean()),'correction_mean':float(correction[i].mean()),'gated_correction_mean':float(gated[i].mean())}for i,s in enumerate(source['scales'])]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    base=Path('artifacts/response_full_20260907')
    inputs={'spiral':base/'population_oracle_spiral_fixture_v2/fixture_arrays.npz','tube':base/'population_oracle_exp_fixture/fixture_arrays.npz','crescent':base/'population_oracle_ve_crescent256/arrays.npz'}
    report={'schema':'covariance_raw is in active raw coefficient coordinates; jacobian_active=covariance_raw/raw_sigma^2 is also the normalized-coordinate Jacobian; all other ambient entries are exactly zero; threshold .25 fixed before results','laws':{}}
    for kind,path in inputs.items():
        report['laws'][kind]=evaluate(path,a.output,kind);print(kind,json.dumps(report['laws'][kind]),flush=True)
    controls=[]
    for u,sigma in [(0,.00442778),(4,.0442778),(6,.0442778)]:
        q=tube_point(u,.7);r=tube_posterior(q,sigma,order=256);c,e,comp,_=gated_correction(r.mean-q,sigma,r.posterior_covariance)
        controls.append({'u':u,'raw_sigma':sigma,'response':r.response,'correction':r.correction,'gated_correction':c,'full':r.full,'gated_full':r.response+c,'eigenvalues':e.tolist(),'components_over_sigma':comp.tolist()})
    report['tube_controls']=controls
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(controls,indent=2))


if __name__=='__main__':main()
