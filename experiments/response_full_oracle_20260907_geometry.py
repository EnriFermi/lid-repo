"""Geometry-assisted controls; these are not label-free deployable readouts."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from experiments.response_full_oracle_20260907 import spiral_curve,spiral_posterior,tube_point,tube_posterior,cylinder_exact


def local_frame(queries,kind):
    if kind=='tube':
        radius=np.linalg.norm(queries[:,1:],axis=1)
        normal=np.column_stack((radius,queries[:,1:]/radius[:,None]))
        return normal/np.linalg.norm(normal,axis=1)[:,None]
    radial=np.linalg.norm(queries,axis=1);t=1/radial
    # Query polar angle is more accurate than t² after float32 normalization.
    sine=queries[:,0]/radial;cosine=queries[:,1]/radial
    tangent=np.column_stack((2*cosine-sine/t**2,-2*sine-cosine/t**2))
    normal=np.column_stack((-tangent[:,1],tangent[:,0]))
    return normal/np.linalg.norm(normal,axis=1)[:,None]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    report={}
    for name,kind,d in [('e1_spiral_pca','spiral',1),('e6_exp_pca','tube',2)]:
        a=dict(np.load(args.input/(name+'_arrays.npz')))
        q=a['queries_raw'];normal=local_frame(q,kind)
        shift=a['posterior_raw']-q[None]
        cnormal=np.einsum('sqi,qi->sq',shift,normal)**2/a['raw_sigma'][:,None]**2
        ctangent=a['correction']-cnormal
        tangent_full=a['response']+ctangent
        half_mean=np.zeros_like(a['posterior_raw'])
        oracle=spiral_posterior if kind=='spiral' else tube_posterior
        for i,sigma in enumerate(a['raw_sigma']):
            for j,point in enumerate(q):half_mean[i,j]=oracle(point,sigma/2,order=128).mean
        first_vector=(4*(half_mean-q[None])-shift)/a['raw_sigma'][:,None,None]
        first_c=np.square(first_vector).sum(axis=-1)
        first_full=a['response']+first_c
        np.savez(args.output/(name+'_geometry.npz'),normal=normal,correction_normal=cnormal,correction_tangent=ctangent,tangent_full=tangent_full,
                 half_posterior_raw=half_mean,first_c=first_c,first_full=first_full,scales=a['scales'],query_ids=a['query_ids'])
        report[name]=[{'scale':float(s),'response_mae':float(np.abs(a['response'][i]-d).mean()),'full_mae':float(np.abs(a['full'][i]-d).mean()),'tangent_full_mae':float(np.abs(tangent_full[i]-d).mean()),
                      'first_full_mae':float(np.abs(first_full[i]-d).mean()),'first_c_mean':float(first_c[i].mean()),
                      'normal_c_mean':float(cnormal[i].mean()),'tangent_c_mean':float(ctangent[i].mean())}for i,s in enumerate(a['scales'])]
    rows=[]
    for u in [0.,2.,4.,6.,7.9,8.]:
        for sigma in [.00442778,.0442778,.14]:
            q=tube_point(u,.7);result=tube_posterior(q,sigma,order=256)
            cylinder=cylinder_exact(q,sigma,np.linalg.norm(q[1:]))
            normal=local_frame(q[None],'tube')[0]
            cnormal=float(((result.mean-q)@normal)**2/sigma**2)
            half=tube_posterior(q,sigma/2,order=256)
            first_c=float(np.square(4*(half.mean-q)-(result.mean-q)).sum()/sigma**2)
            rows.append({'u':u,'raw_sigma':sigma,'radius':float(np.linalg.norm(q[1:])),'radius_over_sigma':float(np.linalg.norm(q[1:])/sigma),
                         'response':result.response,'correction':result.correction,'full':result.full,'normal_c':cnormal,'tangent_full':result.full-cnormal,
                         'first_c':first_c,'first_full':result.response+first_c,
                         'frozen_radius_cylinder_response':cylinder[1],'frozen_radius_cylinder_correction':cylinder[2],'frozen_radius_cylinder_full':cylinder[3]})
    report['tube_counterfactual_grid']=rows
    rows=[]
    for t in [10.,30.,60.,90.]:
        for sigma in [1e-6,.00002,.0002]:
            q=spiral_curve(t);result=spiral_posterior(q,sigma,order=96);local=spiral_posterior(q,sigma,order=96,restrict_turn=True)
            rows.append({'t':t,'raw_sigma':sigma,'turn_gap_approx':np.pi/t**3,'radius':1/t,'response':result.response,'correction':result.correction,'full':result.full,
                         'other_turn_mass':result.metadata['other_turn_mass'],'same_turn_response':local.response,'same_turn_correction':local.correction,
                         'radial_density_C_approx':9*sigma**2*t**2})
    report['spiral_counterfactual_grid']=rows
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
