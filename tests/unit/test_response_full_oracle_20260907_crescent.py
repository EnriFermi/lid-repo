import numpy as np
import pytest
from experiments.response_full_oracle_20260907_crescent import crescent_posterior,normal_interval_moments


def test_disk_center_matches_closed_form_truncated_radial_gaussian():
    radius=3.;sigma=.7
    oracle=crescent_posterior(np.zeros(3),sigma,inner_radius=0)
    upper=radius**2/(2*sigma**2)
    planar=2*(1-upper/np.expm1(upper))
    zm=normal_interval_moments(-3/sigma,3/sigma)
    np.testing.assert_allclose(oracle.response,planar+zm[2]/zm[0],rtol=1e-10,atol=1e-10)
    assert oracle.correction<1e-20


@pytest.mark.parametrize('point,sigma',[(np.array([0.,-2.8,0.]),.03),(np.array([0.,2.998,2.99]),.01),(np.array([2.8,.3,1.]),.3)])
def test_crescent_covariance_is_posterior_jacobian(point,sigma):
    oracle=crescent_posterior(point,sigma)
    step=sigma*1e-4
    jac=np.column_stack([(crescent_posterior(point+step*e,sigma).mean-crescent_posterior(point-step*e,sigma).mean)/(2*step)for e in np.eye(3)])
    np.testing.assert_allclose(jac,oracle.posterior_covariance/sigma**2,rtol=3e-5,atol=3e-6)


def test_deep_volume_point_has_dimension_three_at_small_noise():
    result=crescent_posterior(np.array([0.,-2.8,0.]),.001)
    assert abs(result.response-3)<1e-10
    assert result.correction<1e-20


def test_thin_crescent_and_interval_resolution_are_not_surface_dimension():
    q=np.array([0.,2.998,0.])
    small=crescent_posterior(q,.0001)
    large=crescent_posterior(q,.02)
    assert abs(small.response-3)<1e-5
    assert large.response<2.2
