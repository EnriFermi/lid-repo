import numpy as np
import pytest

from experiments.response_full_oracle_20260907 import (
    spiral_curve,spiral_posterior,tube_point,tube_posterior,cylinder_exact,
)


@pytest.mark.parametrize("u,sigma",[(.001,.03),(2.,.01),(4.,.1),(7.999,.02)])
def test_integrated_cylinder_matches_independent_exact_product(u,sigma):
    radius=.6
    point=np.array([u-4,radius*np.sin(.3),radius*np.cos(.3)])
    numerical=tube_posterior(point,sigma,order=256,fixed_radius=radius)
    exact=cylinder_exact(point,sigma,radius)
    np.testing.assert_allclose(numerical.mean,exact[0],rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose([numerical.response,numerical.correction,numerical.full],exact[1:],rtol=1e-9,atol=1e-10)


@pytest.mark.parametrize("oracle,point,sigma",[(spiral_posterior,spiral_curve(10.),.001),
    (tube_posterior,tube_point(3.,.8),.02)])
def test_posterior_covariance_response_matches_mean_finite_difference(oracle,point,sigma):
    result=oracle(point,sigma,order=128)
    h=1e-4*sigma
    eye=np.eye(len(point))
    jac=np.column_stack([(oracle(point+h*e,sigma,order=128).mean-oracle(point-h*e,sigma,order=128).mean)/(2*h) for e in eye])
    np.testing.assert_allclose(jac,result.posterior_covariance/sigma**2,rtol=2e-5,atol=2e-6)
    np.testing.assert_allclose(np.trace(jac),result.response,rtol=1e-5,atol=1e-6)


def test_spiral_multi_turn_mass_and_order_convergence():
    point=spiral_curve(60.)
    low=spiral_posterior(point,.0002,order=32)
    high=spiral_posterior(point,.0002,order=96)
    local=spiral_posterior(point,.0002,order=96,restrict_turn=True)
    np.testing.assert_allclose([low.response,low.correction],[high.response,high.correction],rtol=2e-6,atol=2e-7)
    assert high.metadata['other_turn_mass']>.8
    assert high.response>1.8
    assert abs(local.response-1)<.03


def test_exact_tube_generator_support_and_small_scale_dimension():
    point=tube_point(2.,.7)
    result=tube_posterior(point,.0001,order=256)
    assert abs(result.response-2)<1e-5
    assert result.correction>0
    assert np.linalg.norm(result.mean-point)<1e-6


def test_spiral_coarse_scale_angular_localization_is_converged():
    point=spiral_curve(15.083708174061021)
    low=spiral_posterior(point,.00577196006842447,order=96)
    high=spiral_posterior(point,.00577196006842447,order=192)
    np.testing.assert_allclose([low.response,low.correction],[high.response,high.correction],atol=1e-9,rtol=1e-9)


def test_boundary_and_curved_interior_reverse_full_response_preference():
    boundary=tube_posterior(tube_point(0.,.7),.00442778,order=256)
    interior=tube_posterior(tube_point(4.,.7),.0442778,order=256)
    assert abs(boundary.full-2)<.002
    assert abs(boundary.response-2)>.6
    assert abs(interior.full-2)>.2
    assert abs(interior.response-2)<.03


def test_spiral_metric_response_matches_chain_rule_for_affine_renderer():
    point=spiral_curve(10.)
    matrix=np.array([[1.,.1],[0.,.8]])
    metric=matrix@matrix.T;sigma=.001
    result=spiral_posterior(point,sigma,metric=metric,order=128)
    h=1e-4*sigma
    jac=np.column_stack([(spiral_posterior(point+h*e,sigma,metric=metric,order=128).mean-spiral_posterior(point-h*e,sigma,metric=metric,order=128).mean)/(2*h)for e in np.eye(2)])
    np.testing.assert_allclose(jac,result.posterior_covariance@metric/sigma**2,rtol=2e-5,atol=2e-6)
    np.testing.assert_allclose(np.trace(jac),result.response,rtol=1e-5,atol=1e-6)


def test_two_scale_correction_worsens_every_positive_flat_boundary_distance():
    from experiments.response_full_oracle_20260907_half import halfline_readouts
    distances=np.linspace(0,8,801);a=halfline_readouts(distances)
    assert np.max(a['first_c']-a['correction'])<1e-15
    assert np.max(a['full']-1)<1e-15
    assert np.max(a['first_full']-a['full'])<1e-15
    assert abs(a['full'][0]-1)<1e-15
    assert abs(a['first_full'][0]-1)<1e-15
    assert np.max(a['full']-a['first_full'])>.2
