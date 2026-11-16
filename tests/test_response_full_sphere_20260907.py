import numpy as np

from experiments.response_full_sphere_20260907 import sphere_component, mixture_posterior


def test_center_is_uniform_sphere():
    center=np.array([2., -1., 3.])
    b,c,_=sphere_component(center, .2, center, 1.7)
    np.testing.assert_allclose(b, center)
    np.testing.assert_allclose(c, np.eye(3)*1.7**2/3)


def test_circle_matches_direct_angular_quadrature():
    angle=np.linspace(0,2*np.pi,32768,endpoint=False)
    values=np.stack([np.cos(angle),np.sin(angle)],-1)*1.3+np.array([.2,-.4])
    query=np.array([.8,.1]); sigma=.37
    weights=np.exp(-((values-query)**2).sum(-1)/(2*sigma*sigma)); weights/=weights.sum()
    mean=(weights[:,None]*values).sum(0)
    cov=np.einsum('n,ni,nj->ij',weights,values-mean,values-mean)
    b,c,_=sphere_component(query,sigma,np.array([.2,-.4]),1.3)
    np.testing.assert_allclose(b,mean,atol=2e-14)
    np.testing.assert_allclose(c,cov,atol=2e-14)


def test_mixture_jacobian_is_covariance_and_log_score():
    centers=np.array([[1.,0.,0.],[-1.,1.,.3]])
    radii=np.array([.7,.3]); query=np.array([.2,.3,.5]); sigma=.6
    out=mixture_posterior(query,sigma,centers,radii)
    eps=1e-5; columns=[]; gradient=[]
    for direction in np.eye(3):
        plus=mixture_posterior(query+eps*direction,sigma,centers,radii)
        minus=mixture_posterior(query-eps*direction,sigma,centers,radii)
        columns.append((plus['posterior']-minus['posterior'])/(2*eps))
        gradient.append((plus['log_evidence_without_gaussian_constant']-minus['log_evidence_without_gaussian_constant'])/(2*eps))
    np.testing.assert_allclose(np.stack(columns,-1),out['covariance']/sigma**2,atol=2e-9)
    np.testing.assert_allclose(gradient,(out['posterior']-query)/sigma**2,atol=2e-9)


def test_s2_full_exact_coth_identity():
    for k in [1., 3., 10., 1000.]:
        out=mixture_posterior(np.array([1.,0.,0.]),k**-.5,[np.zeros(3)],[1.])
        expected=2-4*k/np.expm1(2*k) if k<100 else 2.
        np.testing.assert_allclose(out['full'],expected,atol=2e-10)


def test_batched_single_sphere_translation_invariance():
    query=np.array([[1.,.2,0.],[-.2,.4,.8]])
    shift=np.array([3.,-4.,1.]); sigma=.23
    a=mixture_posterior(query,sigma,[np.zeros(3)],[1.])
    b=mixture_posterior(query+shift,sigma,[shift],[1.])
    np.testing.assert_allclose(a['posterior']+shift,b['posterior'],atol=1e-13)
    np.testing.assert_allclose(a['covariance'],b['covariance'],atol=1e-13)
    np.testing.assert_allclose(a['full'],b['full'],atol=1e-13)
