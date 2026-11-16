import numpy as np
import pytest
from experiments.response_full_oracle_20260907_spaghetti import harmonic_curve,spaghetti_posterior


def test_sine_scalar_population_at_center_matches_dense_independent_integral():
    sigma=.3;result=spaghetti_posterior(np.zeros(1),sigma,harmonics=np.array([1]))
    theta=np.linspace(0,2*np.pi,200000,endpoint=False);x=np.sin(theta);weight=np.exp(-x*x/(2*sigma**2));weight/=weight.sum()
    variance=weight@(x*x)-(weight@x)**2
    np.testing.assert_allclose(result.response,variance/sigma**2,atol=1e-10,rtol=1e-10)
    assert result.correction<1e-20


@pytest.mark.parametrize('theta,sigma',[(.71,.005),(.02,.2),(2.1,.7)])
def test_spaghetti_orders_and_gaussian_tail_are_converged(theta,sigma):
    q=harmonic_curve(theta)
    a=spaghetti_posterior(q,sigma,order=8);b=spaghetti_posterior(q,sigma,order=24,cutoff=14)
    np.testing.assert_allclose([a.response,a.correction],[b.response,b.correction],atol=1e-9,rtol=1e-9)


def test_spaghetti_small_noise_and_covariance_directional_derivative():
    q=harmonic_curve(.71);sigma=.005;r=spaghetti_posterior(q,sigma)
    assert abs(r.response-1)<.001
    direction=np.cos(np.arange(20));direction/=np.linalg.norm(direction);step=1e-4*sigma
    derivative=(spaghetti_posterior(q+step*direction,sigma).mean-spaghetti_posterior(q-step*direction,sigma).mean)/(2*step)
    np.testing.assert_allclose(derivative,r.posterior_covariance@direction/sigma**2,atol=1e-6,rtol=2e-5)


def test_fixed_antithetic_jitter_derivative_is_average_posterior_jacobian():
    point=harmonic_curve(.71);sigma=.03
    noise=np.random.default_rng(45).normal(size=(2,20));noise=np.concatenate([noise,-noise])
    direction=np.sin(np.arange(20));direction/=np.linalg.norm(direction)
    def average(q):return np.mean([spaghetti_posterior(q+sigma*z,sigma).mean for z in noise],axis=0)
    expected=np.mean([spaghetti_posterior(point+sigma*z,sigma).posterior_covariance@direction/sigma**2 for z in noise],axis=0)
    step=1e-4*sigma;actual=(average(point+step*direction)-average(point-step*direction))/(2*step)
    np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=1e-6)
