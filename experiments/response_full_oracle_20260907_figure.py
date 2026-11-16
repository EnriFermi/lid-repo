"""Exact tube population map; geometry and fixed oracle, no trained network."""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.response_full_oracle_20260907 import tube_point,tube_posterior


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    positions=np.linspace(0,8,161);scales=np.geomspace(.0005,1.,61)
    shape=(len(scales),len(positions));response=np.zeros(shape);correction=np.zeros(shape);first=np.zeros(shape);order_error=np.zeros(shape)
    for i,sigma in enumerate(scales):
        for j,u in enumerate(positions):
            q=tube_point(u,.7);result=tube_posterior(q,sigma,order=256);half=tube_posterior(q,sigma/2,order=256)
            low=tube_posterior(q,sigma,order=128)
            response[i,j]=result.response;correction[i,j]=result.correction
            first[i,j]=np.square(4*(half.mean-q)-(result.mean-q)).sum()/sigma**2
            order_error[i,j]=max(abs(result.response-low.response),abs(result.correction-low.correction))
    full=response+correction;first_full=response+first
    np.savez(a.output/'arrays.npz',positions=positions,raw_sigma=scales,response=response,correction=correction,full=full,first_c=first,first_full=first_full,order_error=order_error)
    fig,axes=plt.subplots(1,4,figsize=(17,4),sharex=True,sharey=True,constrained_layout=True)
    fields=[response-2,full-2,first_full-2,np.abs(response-2)-np.abs(full-2)]
    titles=['Response − 2','Full − 2','Two-scale correction + response − 2','Benefit of full over response']
    first_mesh=None
    for ax,field,title in zip(axes,fields,titles):
        mesh=ax.pcolormesh(positions,scales,field,shading='nearest',cmap='RdBu_r' if ax!=axes[-1] else 'RdBu',vmin=-1,vmax=1,rasterized=True)
        ax.plot(positions,3*np.exp(-positions),'k--',lw=1,label='noise = circular radius')
        ax.set_title(title,fontsize=10);ax.set_yscale('log');ax.set_ylim(scales[0],scales[-1]);ax.set_xlabel('Axial coordinate u')
        ax.set_xticks([0,2,4,6,8])
        if first_mesh is None:first_mesh=mesh
    axes[0].set_ylabel('Raw Gaussian noise σ');axes[0].legend(fontsize=7,loc='upper left')
    fig.colorbar(first_mesh,ax=list(axes[:3]),label='Signed dimension error (clipped at ±1)',fraction=.025)
    fig.colorbar(mesh,ax=axes[-1],label='Absolute-error improvement (>0 is better)',fraction=.06)
    fig.suptitle('Exact population posterior on the benchmark shrinking tube (local dimension 2)',fontsize=12)
    fig.savefig(a.output/'tube_geometry_map.png',dpi=200);fig.savefig(a.output/'tube_geometry_map.pdf');plt.close(fig)
    summary={'generator':'u~Uniform[0,8], theta~Uniform[0,2pi], x=(u-4,3exp(-u)sin(theta),3exp(-u)cos(theta))',
             'true_local_dimension':2,'grid_shape':list(shape),'max_128_to_256_order_error':float(order_error.max()),
             'formula_first':'||4*(b_sigma/2(x)-x)-(b_sigma(x)-x)||^2 / sigma^2',
             'scope':'Exact population oracle, not a deployable-model result; two-scale expression is extrapolation outside small-noise asymptotics.',
             'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
