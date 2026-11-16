"""Vector diagrams of posterior spread versus query-to-mean displacement.

The diagrams are exact one-dimensional component controls. Add one free axial
response for the cylinder comparison; do not factorize the actual Exp law.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

from experiments.response_full_oracle_20260907 import bessel_ratio_and_complement, tube_point, tube_posterior


BLUE='#356a91'
RED='#b7433f'
GREEN='#2d7b62'
INK='#202a35'
PALE='#e9f1f6'

LABELS={
    'en':{
        'title':'A correct posterior mean can move away from a clean query',
        'left':'Boundary: one-sided support',
        'right':'Curved interior: a circular section',
        'weight':'Posterior weights',
        'spread':'Spread around b',
        'missing':'No clean points',
        'shift':'Shift from x to b',
        'left_result':'Correction restores the missing boundary response.',
        'right_result':'Correction turns a small underestimate into an overshoot.',
        'caption':'Exact 1D component controls (d = 1), not learned networks.  One independent free coordinate gives the 2D product controls below.',
        'total_left':'With one free coordinate: R = 1.3634,  C = 0.6366,  F = 2.0000  (d = 2)',
        'total_right':'With one axial coordinate: R = 1.9747,  C = 0.2392,  F = 2.2139  (d = 2)',
    },
}


def arrow(ax,start,end,color,*,style='<->',width=1.8,size=10,zorder=5):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle=style,mutation_scale=size,
                                color=color,lw=width,zorder=zorder))


def boundary_panel(ax,labels):
    x=np.linspace(0,3.1,500)
    density=np.sqrt(2/np.pi)*np.exp(-x*x/2)
    mean=np.sqrt(2/np.pi);response=1-2/np.pi;deviation=np.sqrt(response)
    ax.axvspan(-.42,0,color='#eeeeee',zorder=0)
    ax.fill_between(x,0,density,color=PALE)
    ax.plot(x,density,color=BLUE,lw=2)
    dots=np.linspace(.035,2.95,33)
    ax.scatter(dots,np.zeros_like(dots),s=18,color=BLUE,alpha=.75,zorder=3)
    ax.plot([0,3.15],[0,0],color=BLUE,lw=1.3)
    ax.plot([mean,mean],[0,.9],ls=':',lw=1,color=RED)
    ax.scatter([0],[0],s=105,marker='*',color=INK,zorder=8)
    ax.scatter([mean],[0],s=45,marker='D',color=RED,zorder=8)
    ax.text(-.04,.095,r'$x$',color=INK,ha='center',fontsize=14)
    ax.text(mean+.06,.095,r'$b$',color=RED,ha='center',fontsize=14)
    ax.text(1.9,.055,r'$X$',color=BLUE,ha='center',fontsize=12)
    ax.text(-.27,.43,labels['missing'],rotation=90,ha='center',va='center',fontsize=9,color='#777777')
    ax.text(2.0,.44,labels['weight'],color=BLUE,fontsize=11)
    # This bar is exactly one posterior standard deviation on either side.
    ax.errorbar([mean],[.88],xerr=deviation,fmt='o',color=GREEN,ecolor=GREEN,capsize=4,lw=1.8,ms=4)
    ax.text(mean,.96,labels['spread'],ha='center',color=GREEN,fontsize=11)
    ax.text(mean,.78,r'$\pm\sqrt{R}\,\sigma$',ha='center',color=GREEN,fontsize=11)
    arrow(ax,(0,-.19),(mean,-.19),RED)
    ax.text(mean/2,-.33,r'$|b-x|$',ha='center',color=RED,fontsize=12)
    arrow(ax,(1.85,-.19),(2.85,-.19),INK,width=1.2)
    ax.text(2.35,-.33,r'$\sigma$',ha='center',color=INK,fontsize=12)
    ax.set(xlim=(-.42,3.2),ylim=(-.48,1.07))
    ax.axis('off')
    return {'response':float(response),'correction':float(mean**2),'full':1.,'mean_over_sigma':float(mean)}


def circle_panel(ax,labels):
    # Exactly the frozen-radius control at benchmark Exp u4, raw sigma .0442778.
    raw_radius=3*np.exp(-4.);raw_sigma=.0442778
    radius=raw_radius/raw_sigma;kappa=radius**2
    ratio,complement=bessel_ratio_and_complement(np.array([kappa]))
    mean=radius*ratio[0]
    response=kappa*complement[0]*(1+ratio[0]);correction=kappa*complement[0]**2
    angles=np.linspace(0,2*np.pi,130,endpoint=False)
    points=radius*np.column_stack((np.cos(angles),np.sin(angles)))
    weights=np.exp(kappa*(np.cos(angles)-1))
    fine=np.linspace(0,2*np.pi,700)
    ax.plot(radius*np.cos(fine),radius*np.sin(fine),color='#b9c8d2',lw=1.0,zorder=1)
    ax.scatter(points[:,0],points[:,1],s=60*weights,color=BLUE,alpha=.88,zorder=3)
    ax.scatter([radius],[0],s=105,marker='*',color=INK,zorder=8)
    ax.scatter([mean],[0],s=45,marker='D',color=RED,zorder=8)
    ax.text(radius+.12,.10,r'$x$',color=INK,ha='left',fontsize=14)
    ax.text(mean-.08,.13,r'$b$',color=RED,ha='center',fontsize=14)
    ax.text(.99,1.0,r'$X$',color=BLUE,ha='center',fontsize=12)
    ax.plot([0],[0],marker='+',ms=5,color='#aaaaaa')
    # Actual support points, not a fictitious Gaussian cloud around the mean.
    for angle in [.85,-.95,2.15]:
        target=radius*np.array([np.cos(angle),np.sin(angle)])
        arrow(ax,(mean,0),target,GREEN,style='->',width=1.2,size=8,zorder=2)
    ax.text(-1.33,1.34,labels['spread'],color=GREEN,fontsize=11)
    ax.text(-1.44,-1.38,labels['weight'],color=BLUE,fontsize=10)
    arrow(ax,(mean,-.19),(radius,-.19),RED)
    ax.text(1.62,-.43,r'$|b-x|$',ha='center',color=RED,fontsize=11)
    arrow(ax,(-.30,-1.68),(.70,-1.68),INK,width=1.2)
    ax.text(.20,-1.87,r'$\sigma$',ha='center',color=INK,fontsize=12)
    ax.text(1.17,1.27,r'$a/\sigma=1.24$',ha='center',fontsize=11,color='#66727b')
    ax.set(xlim=(-1.75,1.95),ylim=(-2.0,1.60),aspect='equal')
    ax.axis('off')
    return {'raw_radius':float(raw_radius),'raw_sigma':raw_sigma,'radius_over_sigma':float(radius),
            'response':float(response),'correction':float(correction),'full':float(response+correction),
            'mean_over_sigma':float(mean)}


def make_figure(output,language):
    labels=LABELS[language]
    fig=plt.figure(figsize=(13.7,6.65),facecolor='white')
    left=fig.add_axes([.055,.325,.40,.46]);right=fig.add_axes([.535,.285,.41,.515])
    b=boundary_panel(left,labels);c=circle_panel(right,labels)
    fig.text(.5,.958,labels['title'],ha='center',fontsize=15,fontweight='normal',color=INK)
    fig.text(.255,.85,'A   '+labels['left'],ha='center',fontsize=12,fontweight='bold',color=INK)
    fig.text(.74,.85,'B   '+labels['right'],ha='center',fontsize=12,fontweight='bold',color=INK)
    fig.text(.28,.903,r'$R=\mathbb{E}[\|X-b\|^2\mid Y=x]/\sigma^2$',ha='center',fontsize=13,color=GREEN)
    fig.text(.61,.903,r'$C=\|b-x\|^2/\sigma^2$',ha='center',fontsize=13,color=RED)
    fig.text(.855,.903,r'$F=R+C$',ha='center',fontsize=13,color=INK)
    for center,data,key in [(.255,b,'left_result'),(.74,c,'right_result')]:
        numeric=rf'$R={data["response"]:.4f}\quad C={data["correction"]:.4f}\quad F={data["full"]:.4f}$'
        fig.text(center,.245,numeric,ha='center',fontsize=13,color=INK,
                 bbox={'boxstyle':'round,pad=.5','facecolor':'#f4f7f9','edgecolor':'#e0e7ec'})
        fig.text(center,.18,labels[key],ha='center',fontsize=10.5,color=INK)
    fig.text(.5,.105,labels['caption'],ha='center',fontsize=9.5,color='#4e5b65')
    fig.text(.255,.06,labels['total_left'],ha='center',fontsize=9,color='#4e5b65')
    fig.text(.74,.06,labels['total_right'],ha='center',fontsize=9,color='#4e5b65')
    fig.savefig(output/f'posterior_mechanism_{language}.pdf',bbox_inches='tight',pad_inches=.12)
    fig.savefig(output/f'posterior_mechanism_{language}.png',dpi=200,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    return {'boundary_component':b,'circle_component':c}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    values={language:make_figure(args.output,language)for language in ['en']}
    controls=[]
    for u,sigma in [(0.,.00442778),(4.,.0442778),(6.,.0442778)]:
        result=tube_posterior(tube_point(u,.7),sigma,order=256)
        controls.append({'u':u,'raw_sigma':sigma,'response':result.response,'correction':result.correction,'full':result.full})
    report={'scope':'Exact one-dimensional local half-line and uniform-circle controls. The actual Exp law is coupled and is not factorized. One free/axial coordinate adds1 toR andd.',
            'displayed_values':values['en'],'actual_exp_reference_points':controls,
            'posterior_points':'Circle dot area encodes relative angular posterior weight; boundary curve is normalized half-normal density. No sampled neural model or generated raster artwork.'}
    (args.output/'values.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
