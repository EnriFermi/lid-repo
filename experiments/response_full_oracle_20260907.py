"""Population posterior references for the exact Spiral and Exp benchmark laws.

No finite sample KDE is used. Spiral integrates its exact uniform latent t;
Exp integrates its exact uniform axial latent after analytic angular averaging.
Gaussian posterior response is tr(Cov[X|query])/sigma^2. All calculations
use float64 and centered moments, with independent quadrature-order checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from scipy.special import i0e, i1e, ndtr, roots_legendre


@dataclass(frozen=True)
class OracleResult:
    mean: np.ndarray
    response: float
    correction: float
    full: float
    posterior_covariance: np.ndarray
    log_mass: float
    metadata: dict


@lru_cache(maxsize=12)
def rule(order):
    if order < 8:raise ValueError("Quadrature order must be >=8")
    return roots_legendre(order)


def _moments(query, displacement, log_weights, *, metric=None, conditional_covariance=None, metadata=None):
    query=np.asarray(query,dtype=np.float64)
    if metric is None:metric=np.eye(len(query))
    metric=np.asarray(metric,dtype=np.float64)
    maximum=float(np.max(log_weights))
    weights=np.exp(log_weights-maximum)
    mass=float(weights.sum())
    if not np.isfinite(mass) or mass<=0:raise FloatingPointError("No posterior quadrature mass")
    weights/=mass
    shift=weights@displacement
    centered=displacement-shift
    covariance=(centered*weights[:,None]).T@centered
    if conditional_covariance is not None:
        covariance+=np.einsum("n,nij->ij",weights,conditional_covariance)
    metadata={} if metadata is None else dict(metadata)
    return shift,covariance,maximum+math.log(mass),weights,metadata


def spiral_curve(t):
    t=np.asarray(t,dtype=np.float64)
    return np.stack((np.sin(t*t)/t,np.cos(t*t)/t),axis=-1)


def spiral_posterior(query,sigma,*,order=48,cutoff=10.,metric=None,restrict_turn=False,t_bounds=(1.,100.)):
    """Single-query exact-law integral, localized by a Gaussian tail envelope.

    Integrate phase=t² in angle-centered turn cells. Restricting to the query's
    nearest turn is a geometry intervention, not the original population.
    metric supports the measured Gram matrix of a nearly-isometric renderer.
    """
    q=np.asarray(query,dtype=np.float64)
    if q.shape!=(2,) or sigma<=0:raise ValueError("Require 2D query and positive sigma")
    metric=np.eye(2) if metric is None else np.asarray(metric,dtype=np.float64)
    smallest=float(np.linalg.eigvalsh(metric).min())
    if smallest<=0:raise ValueError("Metric must be positive definite")
    sigma_bound=sigma/math.sqrt(smallest)
    radius=float(np.linalg.norm(q))
    lo_t,hi_t=map(float,t_bounds)
    if not 0<lo_t<hi_t:raise ValueError("Invalid latent support")
    radial_low=max(1/hi_t,radius-cutoff*sigma_bound)
    radial_high=min(1/lo_t,radius+cutoff*sigma_bound)
    if radial_high<radial_low:
        # Off-support far queries are not the primary use; retain the whole
        # support rather than return an empty approximation.
        radial_low,radial_high=1/hi_t,1/lo_t
    phase_low=max(lo_t**2,1/radial_high**2)
    phase_high=min(hi_t**2,1/radial_low**2)
    angle=float(np.arctan2(q[0],q[1])) if radius else 0.
    first=math.ceil((phase_low-angle-math.pi)/(2*math.pi))
    last=math.floor((phase_high-angle+math.pi)/(2*math.pi))
    turns=np.arange(first,last+1,dtype=np.int64)
    centers=angle+2*math.pi*turns
    # A separate radius lower bound for each phase cell avoids integrating
    # narrow high-concentration angular peaks across an unnecessarily full
    # [-pi,pi] interval when a distant inner turn expands the global envelope.
    cell_radial_low=1/np.sqrt(np.maximum(phase_low,np.minimum(phase_high,centers+math.pi)))
    if radius>0:
        angular_bound=2*np.arcsin(np.minimum(1.,cutoff*sigma_bound/(2*np.sqrt(radius*cell_radial_low))))
    else:angular_bound=np.full_like(centers,math.pi)
    low=np.maximum(phase_low,centers-angular_bound)
    high=np.minimum(phase_high,centers+angular_bound)
    query_phase=1/radius**2 if radius else hi_t**2
    nearest_turn=int(np.rint((query_phase-angle)/(2*math.pi)))
    mask=high>low
    if restrict_turn:mask&=(turns==nearest_turn)
    low,high,turns=low[mask],high[mask],turns[mask]
    if not len(low):raise FloatingPointError("No retained phase interval")
    nodes,weights=rule(order)
    half=(high-low)/2
    phase=((high+low)/2)[:,None]+half[:,None]*nodes
    t=np.sqrt(phase)
    samples=np.stack((np.sin(phase)/t,np.cos(phase)/t),axis=-1).reshape(-1,2)
    displacement=samples-q
    squared=np.einsum("ni,ij,nj->n",displacement,metric,displacement)
    quadrature=(half[:,None]*weights[None,:]/(2*t)).reshape(-1)
    log_weights=np.log(quadrature)-squared/(2*sigma*sigma)
    shift,covariance,log_mass,probabilities,metadata=_moments(q,displacement,log_weights,metric=metric)
    node_turns=np.repeat(turns,order)
    same_turn=float(probabilities[node_turns==nearest_turn].sum())
    response=float(np.trace(metric@covariance)/sigma**2)
    correction=float(shift@metric@shift/sigma**2)
    metadata.update(order=order,cutoff=cutoff,quadrature_nodes=len(displacement),turn_count=len(turns),
                    nearest_turn=nearest_turn,same_turn_mass=same_turn,
                    other_turn_mass=1-same_turn,angular_halfwidth=float(np.max(angular_bound)),
                    latent_t_inferred=math.sqrt(query_phase),restricted_turn=restrict_turn)
    return OracleResult(q+shift,response,correction,response+correction,covariance,
                        log_mass-math.log(hi_t-lo_t),metadata)


def bessel_ratio_and_complement(kappa):
    k=np.asarray(kappa,dtype=np.float64)
    ratio=i1e(k)/i0e(k)
    complement=1-ratio
    large=k>=50
    if np.any(large):
        inverse=1/k[large]
        coefficients=(1/2,1/8,1/8,25/128,13/32,1073/1024,103/32,375733/32768,23797/512,55384775/262144)
        small=np.zeros_like(inverse)
        for power,coefficient in enumerate(coefficients,start=1):small+=coefficient*inverse**power
        complement=np.array(complement,copy=True)
        ratio=np.array(ratio,copy=True)
        complement[large]=small
        ratio[large]=1-small
    return ratio,complement


def tube_point(u,theta):
    u,theta=np.broadcast_arrays(np.asarray(u,dtype=np.float64),np.asarray(theta,dtype=np.float64))
    radius=3*np.exp(-u)
    return np.stack((u-4,radius*np.sin(theta),radius*np.cos(theta)),axis=-1)


def tube_posterior(query,sigma,*,order=128,cutoff=10.,u_bounds=(0.,8.),fixed_radius=None,prior='uniform_u'):
    """Exact angular integration; one-dimensional axial Gauss--Legendre rule.

    fixed_radius produces an explicitly distinct cylinder counterfactual.
    The actual Exp law always uses radius(u)=3 exp(-u).
    """
    q=np.asarray(query,dtype=np.float64)
    if q.shape!=(3,) or sigma<=0:raise ValueError("Require 3D query and positive sigma")
    if prior not in ['uniform_u','uniform_area']:raise ValueError('Unknown tube prior')
    start,end=map(float,u_bounds)
    if not start<end:raise ValueError("Invalid axial support")
    low=max(start,q[0]+4-cutoff*sigma)
    high=min(end,q[0]+4+cutoff*sigma)
    if high<=low:low,high=start,end
    nodes,weights=rule(order)
    half=(high-low)/2
    u=(high+low)/2+half*nodes
    radius=3*np.exp(-u) if fixed_radius is None else np.full_like(u,float(fixed_radius))
    rho=float(np.linalg.norm(q[1:]))
    direction=q[1:]/rho if rho>0 else np.array([1.,0.])
    perpendicular=np.array([-direction[1],direction[0]])
    kappa=radius*rho/sigma**2
    ratio,complement=bessel_ratio_and_complement(kappa)
    log_weights=np.log(half*weights)+np.log(i0e(kappa))-((u-4-q[0])**2+(radius-rho)**2)/(2*sigma**2)
    prior_mass=end-start
    if prior=='uniform_area':
        if fixed_radius is None:
            log_weights+=np.log(radius)+.5*np.log1p(radius**2)
            primitive=lambda r:.5*(r*np.sqrt(1+r*r)+np.arcsinh(r))
            prior_mass=primitive(3*np.exp(-start))-primitive(3*np.exp(-end))
        else:
            log_weights+=np.log(radius)
            prior_mass=(end-start)*fixed_radius
    transverse_shift=(radius-rho)-radius*complement
    displacement=np.concatenate(((u-4-q[0])[:,None],transverse_shift[:,None]*direction[None,:]),axis=1)
    # Angular covariance of a von Mises circle. The trace is stable even at
    # high concentration; tangential variance is r²*A/k, radial is the rest.
    total_angular_variance=radius**2*complement*(1+ratio)
    tangential_variance=np.divide(radius**2*ratio,kappa,out=radius**2/2,where=kappa>0)
    radial_variance=np.maximum(0.,total_angular_variance-tangential_variance)
    conditional=np.zeros((len(u),3,3),dtype=np.float64)
    conditional[:,1:,1:]=(radial_variance[:,None,None]*np.outer(direction,direction)
                          +tangential_variance[:,None,None]*np.outer(perpendicular,perpendicular))
    shift,covariance,log_mass,probabilities,metadata=_moments(q,displacement,log_weights,conditional_covariance=conditional)
    response=float(np.trace(covariance)/sigma**2)
    correction=float(shift@shift/sigma**2)
    metadata.update(order=order,cutoff=cutoff,quadrature_nodes=len(u),
                    axial_posterior_mean=float(probabilities@u),
                    axial_posterior_variance=float(probabilities@(u-probabilities@u)**2),
                    expected_radius=float(probabilities@radius),fixed_radius=fixed_radius,prior=prior,
                    conditional_angular_variance=float(probabilities@total_angular_variance)/sigma**2)
    return OracleResult(q+shift,response,correction,response+correction,covariance,
                        log_mass-math.log(prior_mass),metadata)


def cylinder_exact(query,sigma,radius,*,u_bounds=(0.,8.)):
    """Independent closed-form axial-uniform x circle cross-check."""
    q=np.asarray(query,dtype=np.float64)
    a=(u_bounds[0]-4-q[0])/sigma
    b=(u_bounds[1]-4-q[0])/sigma
    phi=lambda x:np.exp(-x*x/2)/np.sqrt(2*np.pi)
    mass=ndtr(b)-ndtr(a)
    shift=(phi(a)-phi(b))/mass
    variance=1+(a*phi(a)-b*phi(b))/mass-shift**2
    rho=np.linalg.norm(q[1:]);kappa=radius*rho/sigma**2
    ratio,complement=bessel_ratio_and_complement(np.array([kappa]))
    direction=q[1:]/rho if rho else np.array([1.,0.])
    mean=np.concatenate(([q[0]+sigma*shift],radius*ratio[0]*direction))
    response=float(variance+radius**2*complement[0]*(1+ratio[0])/sigma**2)
    correction=float(np.square(mean-q).sum()/sigma**2)
    return mean,response,correction,response+correction
