import numpy as np
from experiments.response_full_oracle_20260907 import tube_point,tube_posterior


def test_uniform_area_prior_is_identical_on_fixed_radius_cylinder():
    radius=.6;q=np.array([-.4,radius*np.sin(.7),radius*np.cos(.7)])
    a=tube_posterior(q,.05,fixed_radius=radius,prior='uniform_u')
    b=tube_posterior(q,.05,fixed_radius=radius,prior='uniform_area')
    np.testing.assert_allclose(a.mean,b.mean,atol=1e-14,rtol=1e-14)
    np.testing.assert_allclose([a.response,a.correction,a.log_mass],[b.response,b.correction,b.log_mass],atol=1e-13,rtol=1e-13)


def test_density_counterfactual_shift_has_predicted_small_scale_tangent_direction():
    u=2.;theta=.7;sigma=.0001;radius=3*np.exp(-u);q=tube_point(u,theta)
    a=tube_posterior(q,sigma,order=256,prior='uniform_u')
    b=tube_posterior(q,sigma,order=256,prior='uniform_area')
    derivative=np.array([1.,-radius*np.sin(theta),-radius*np.cos(theta)])
    expected=-(1+2*radius**2)/(1+radius**2)**2*derivative
    np.testing.assert_allclose((b.mean-a.mean)/sigma**2,expected,atol=1e-6,rtol=1e-6)


def test_default_tube_prior_remains_explicit_uniform_u():
    q=tube_point(4.,.7)
    a=tube_posterior(q,.0442778,order=256)
    b=tube_posterior(q,.0442778,order=256,prior='uniform_u')
    assert a.response==b.response and a.correction==b.correction
    np.testing.assert_array_equal(a.mean,b.mean)
