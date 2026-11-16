"""Exact uniform-volume Crescent law: eccentric planar annulus x z interval.

The benchmark's rejection sampler is uniform in the accepted volume. It is
not a two-dimensional moon surface. Integrate x adaptively; positive vertical
intervals and independent z have analytic truncated-Gaussian moments.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.integrate import quad_vec
from scipy.special import ndtr
from experiments.response_full_oracle_20260907 import OracleResult


def normal_interval_moments(a,b):
    """Unnormalized zeroth/first/second moments of standard normal on[a,b]."""
    if b<=a:return np.zeros(3)
    mass=ndtr(-a)-ndtr(-b) if a>=0 else ndtr(b)-ndtr(a)
    pa=math.exp(-a*a/2)/math.sqrt(2*math.pi);pb=math.exp(-b*b/2)/math.sqrt(2*math.pi)
    return np.array([mass,pa-pb,mass+a*pa-b*pb])


def crescent_posterior(query,sigma,*,cutoff=12.,epsabs=1e-12,epsrel=1e-10,outer_radius=3.,inner_radius=2.697,inner_offset=.3,z_bounds=(-3.,3.)):
    q=np.asarray(query,dtype=np.float64)
    if q.shape!=(3,)or sigma<=0:raise ValueError('Require 3D query and positive noise')
    if not (0<=inner_radius<outer_radius and abs(inner_offset)+inner_radius<=outer_radius):
        raise ValueError('This integrator requires a fully contained inner disk')
    if not z_bounds[0]<z_bounds[1]:raise ValueError('Require increasing z bounds')
    low=max(-outer_radius,q[0]-cutoff*sigma);high=min(outer_radius,q[0]+cutoff*sigma)
    if low>=high:raise ValueError('Query too far from planar support')
    def integrand(t):
        x=q[0]+sigma*t
        outer=math.sqrt(max(0.,outer_radius**2-x*x))
        intervals=[(-outer,outer)]
        if inner_radius>0 and abs(x)<inner_radius:
            inner=math.sqrt(max(0.,inner_radius**2-x*x))
            intervals=[(-outer,inner_offset-inner),(inner_offset+inner,outer)]
        moments=np.zeros(3)
        for a,b in intervals:
            moments+=normal_interval_moments((a-q[1])/sigma,(b-q[1])/sigma)
        mass,m1,m2=moments
        return math.exp(-t*t/2)/math.sqrt(2*math.pi)*np.array([mass,t*mass,m1,t*t*mass,t*m1,m2])
    breaks=[(x-q[0])/sigma for x in [-inner_radius,inner_radius] if low<x<high]
    value,error=quad_vec(integrand,(low-q[0])/sigma,(high-q[0])/sigma,epsabs=epsabs,epsrel=epsrel,points=breaks,norm='max',limit=10000)
    mass=value[0]
    if not mass>0:raise FloatingPointError('No planar posterior mass')
    ex,ey=value[1:3]/mass
    centered2=np.array([[value[3]/mass-ex*ex,value[4]/mass-ex*ey],[value[4]/mass-ex*ey,value[5]/mass-ey*ey]])
    zm=normal_interval_moments((z_bounds[0]-q[2])/sigma,(z_bounds[1]-q[2])/sigma)
    ez=zm[1]/zm[0];vz=zm[2]/zm[0]-ez*ez
    shift=sigma*np.array([ex,ey,ez]);cov=np.zeros((3,3));cov[:2,:2]=sigma**2*centered2;cov[2,2]=sigma**2*vz
    response=float(np.trace(cov)/sigma**2);correction=float(shift@shift/sigma**2)
    volume=math.pi*(outer_radius**2-inner_radius**2)*(z_bounds[1]-z_bounds[0])
    log_mass=math.log(mass*zm[0])+1.5*math.log(2*math.pi*sigma**2)-math.log(volume)
    return OracleResult(q+shift,response,correction,response+correction,cov,log_mass,
                        {'cutoff':cutoff,'epsabs':epsabs,'epsrel':epsrel,'unnormalized_quadrature_error_estimate':float(error),
                         'planar_mass':float(mass),'relative_moment_error_estimate':float(error/mass),'z_response':float(vz),
                         'z_correction':float(ez*ez),'true_local_dimension':3})


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--query-count',type=int,default=256);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    source=dict(np.load(a.input));rms=float(source['normalization_rms']);mean=source['normalization_mean'].astype(np.float64)
    normalized=source['queries'][:a.query_count];raw=normalized.astype(np.float64)*rms+mean;q=raw[:,:3];scales=source['scales'];raw_sigma=scales*rms
    shape=(len(scales),len(q));response=np.zeros(shape);correction=np.zeros(shape);first_c=np.zeros(shape);integration_error=np.zeros(shape);check_error=np.zeros(shape)
    posterior=np.zeros((len(scales),len(q),30));half=np.zeros_like(posterior)
    for i,sigma in enumerate(raw_sigma):
        for j,point in enumerate(q):
            result=crescent_posterior(point,sigma);small=crescent_posterior(point,sigma/2)
            response[i,j]=result.response;correction[i,j]=result.correction;posterior[i,j,:3]=result.mean;half[i,j,:3]=small.mean
            first_c[i,j]=np.square(4*(small.mean-point)-(result.mean-point)).sum()/sigma**2
            integration_error[i,j]=max(result.metadata['relative_moment_error_estimate'],small.metadata['relative_moment_error_estimate'])
            # Independent stricter tolerance and wider Gaussian envelope on
            # every eighth query; per-query adaptive error estimates on all.
            if j%8==0:
                strict=crescent_posterior(point,sigma,cutoff=14,epsabs=1e-14,epsrel=1e-12)
                check_error[i,j]=max(abs(strict.response-result.response),abs(strict.correction-result.correction))
        print(float(scales[i]),float(response[i].mean()),float(correction[i].mean()),flush=True)
    full=response+correction;first_full=response+first_c
    np.savez(a.output/'arrays.npz',posterior=(posterior-mean)/rms,posterior_half=(half-mean)/rms,response=response,correction=correction,full=full,
             first_c=first_c,first_full=first_full,scales=scales,half_scales=scales/2,raw_sigma=raw_sigma,queries=normalized,queries_raw=q,
             query_ids=source['query_ids'][:len(q)],normalization_mean=mean,normalization_scale=np.array(rms),integration_error=integration_error,strict_check_error=check_error)
    report={'source':str(a.input),'source_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),'query_count':len(q),'true_local_dimension':3,
            'max_adaptive_relative_moment_error_estimate':float(integration_error.max()),'max_strict_tolerance_cutoff_readout_difference':float(check_error.max()),
            'schema':'posterior and posterior_half are [scale,query,30] in source normalized data coordinates; half is at scales/2; original response used in first_full',
            'summary':[{'scale':float(s),'response_mae':float(np.abs(response[i]-3).mean()),'full_mae':float(np.abs(full[i]-3).mean()),
                         'first_full_mae':float(np.abs(first_full[i]-3).mean()),'response_mean':float(response[i].mean()),'correction_mean':float(correction[i].mean())}for i,s in enumerate(scales)]}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
