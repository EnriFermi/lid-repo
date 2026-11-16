"""Exact uniform-phase population posterior of the twenty-harmonic curve.

Global Lipschitz bounds prune distant phase cells. Retained cells are refined
until their maximum displacement is a small multiple of sigma, then integrated
by Gauss--Legendre rules. This resolves thin likelihood peaks without a KDE.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from experiments.response_full_oracle_20260907 import OracleResult,rule,_moments


def harmonic_curve(theta,harmonics=np.arange(2,22)):
    return np.sin(np.asarray(theta)[...,None]*np.asarray(harmonics))


def spaghetti_posterior(query,sigma,*,order=16,cutoff=12.,initial_cells=512,max_cell_displacement=2.,harmonics=np.arange(2,22),query_phase=None):
    q=np.asarray(query,dtype=np.float64);harmonics=np.asarray(harmonics,dtype=np.float64)
    if q.shape!=(len(harmonics),)or sigma<=0:raise ValueError('Query dimension must match harmonics; positive noise required')
    if initial_cells<4 or max_cell_displacement<=0:raise ValueError('Invalid phase partition')
    lipschitz=float(np.linalg.norm(harmonics))
    edges=np.linspace(0,2*np.pi,initial_cells+1);mid=(edges[:-1]+edges[1:])/2;half=(edges[1]-edges[0])/2
    distance=np.linalg.norm(harmonic_curve(mid,harmonics)-q,axis=1)
    keep=distance<=cutoff*sigma+lipschitz*half
    parent=np.flatnonzero(keep)
    if not len(parent):raise FloatingPointError('No retained phase cell')
    subdivision=max(1,int(np.ceil(2*half*lipschitz/(max_cell_displacement*sigma))))
    small_width=2*half/subdivision
    low=(edges[parent,None]+small_width*np.arange(subdivision)[None,:]).reshape(-1)
    mid=low+small_width/2
    # A second conservative exclusion keeps the computational cost small at
    # very low noise while retaining every point within cutoff*sigma.
    distance=np.linalg.norm(harmonic_curve(mid,harmonics)-q,axis=1)
    mask=distance<=cutoff*sigma+lipschitz*small_width/2
    mid=mid[mask]
    nodes,weights=rule(order)
    phase=(mid[:,None]+small_width/2*nodes[None,:]).reshape(-1)
    points=harmonic_curve(phase,harmonics);displacement=points-q
    log_weights=np.tile(np.log(small_width/2*weights),len(mid))-np.square(displacement).sum(axis=1)/(2*sigma**2)
    shift,covariance,log_mass,probabilities,metadata=_moments(q,displacement,log_weights)
    response=float(np.trace(covariance)/sigma**2);correction=float(shift@shift/sigma**2)
    metadata.update(order=order,cutoff=cutoff,initial_cells=initial_cells,parent_cells=len(parent),retained_refined_cells=len(mid),
                    quadrature_nodes=len(phase),global_lipschitz=lipschitz,max_cell_displacement=max_cell_displacement)
    if query_phase is not None:
        angular_distance=np.abs((phase-query_phase+np.pi)%(2*np.pi)-np.pi)
        metadata['posterior_phase_distance_mean']=float(probabilities@angular_distance)
        metadata['remote_phase_mass_over_pi_over_21']=float(probabilities[angular_distance>np.pi/21].sum())
    return OracleResult(q+shift,response,correction,response+correction,covariance,log_mass-math.log(2*np.pi),metadata)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--query-count',type=int,default=64);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    source=dict(np.load(a.input));mean=source['normalization_mean'].astype(np.float64);rms=float(source['normalization_rms']);normalized=source['queries'][:a.query_count]
    q=(normalized.astype(np.float64)*rms+mean)[:,:20];scales=source['scales'];sigmas=scales*rms;ids=source['query_ids'][:len(q)]
    all_phase=np.random.RandomState(0).uniform(0,2*np.pi,102000);phases=all_phase[101000+ids]
    shape=(len(scales),len(q));response=np.zeros(shape);correction=np.zeros(shape);first=np.zeros(shape);errors=np.zeros(shape);cutoff_errors=np.zeros(shape);remote=np.zeros(shape);nodes=np.zeros(shape,dtype=np.int64)
    posterior=np.zeros((len(scales),len(q),30));half=np.zeros_like(posterior);covariance=np.zeros(shape+(20,20))
    for i,sigma in enumerate(sigmas):
        for j,query in enumerate(q):
            high=spaghetti_posterior(query,sigma,query_phase=phases[j]);low=spaghetti_posterior(query,sigma,order=8);wide=spaghetti_posterior(query,sigma,cutoff=14.)
            small=spaghetti_posterior(query,sigma/2)
            posterior[i,j,:20]=high.mean;half[i,j,:20]=small.mean;covariance[i,j]=high.posterior_covariance
            response[i,j]=high.response;correction[i,j]=high.correction
            first[i,j]=np.square(4*(small.mean-query)-(high.mean-query)).sum()/sigma**2
            errors[i,j]=max(abs(high.response-low.response),abs(high.correction-low.correction))
            cutoff_errors[i,j]=max(abs(high.response-wide.response),abs(high.correction-wide.correction))
            remote[i,j]=high.metadata['remote_phase_mass_over_pi_over_21'];nodes[i,j]=high.metadata['quadrature_nodes']
        print(float(scales[i]),float(response[i].mean()),float(correction[i].mean()),flush=True)
    full=response+correction;first_full=response+first
    np.savez(a.output/'arrays.npz',posterior=(posterior-mean)/rms,posterior_half=(half-mean)/rms,response=response,correction=correction,full=full,first_c=first,first_full=first_full,
             covariance_raw=covariance,jacobian_active=covariance/sigmas[:,None,None,None]**2,query_ids=ids,query_phase=phases,queries=normalized,queries_raw=q,scales=scales,half_scales=scales/2,raw_sigma=sigmas,
             normalization_mean=mean,normalization_scale=np.array(rms),order_error=errors,cutoff_error=cutoff_errors,remote_phase_mass=remote,quadrature_nodes=nodes)
    report={'source':str(a.input),'query_count':len(q),'true_geometric_dimension':1,'upstream_label':20,'upstream_label_is_not_geometric_rank':True,
            'generator':'theta uniform[0,2pi]; active coordinates sin(k theta), k=2,...,21; remaining10 coordinates zero',
            'max_8_to_16_order_error':float(errors.max()),'max_12_to_14_cutoff_error':float(cutoff_errors.max()),
            'phase_reconstruction_max_error':float(np.abs(harmonic_curve(phases)-q).max()),
            'summary':[{'scale':float(s),'response_mae':float(np.abs(response[i]-1).mean()),'full_mae':float(np.abs(full[i]-1).mean()),'first_full_mae':float(np.abs(first_full[i]-1).mean()),
                         'response_mean':float(response[i].mean()),'correction_mean':float(correction[i].mean()),'remote_phase_mass_mean':float(remote[i].mean())}for i,s in enumerate(scales)]}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
