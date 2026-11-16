import numpy as np
from experiments.response_full_temporal_covariance_terms_20260907 import covariance_terms


def test_true_shift_only_linear_covariance_can_be_negative():
    q=np.zeros((1,1));oracle=np.ones((1,1));motion=np.array([-1.,0.,1.])[:,None,None]*.1
    row,a=covariance_terms(oracle[None]+motion,-motion[:,:,0],q,oracle,1.)
    assert row['cov_linear_true_shift']<0
    np.testing.assert_allclose(row['cov_linear_mean_error'],0,atol=1e-15)
    np.testing.assert_allclose(row['cov_quadratic_motion'],0,atol=1e-15)


def test_mean_error_shift_not_true_geometry_can_cause_covariance():
    q=oracle=np.zeros((2,3));rng=np.random.default_rng(4)
    b=rng.normal(size=(8,2,3))+2;r=rng.normal(size=(8,2))
    row,a=covariance_terms(b,r,q,oracle,.2)
    assert row['cov_linear_true_shift']==0
    np.testing.assert_allclose(a['covariance'],a['cov_linear_mean_error']+a['cov_quadratic_motion'])


def test_exact_spatial_and_temporal_covariance_sum_with_nonuniform_query_means():
    rng=np.random.default_rng(3);q=rng.normal(size=(4,5));oracle=q+.3
    b=rng.normal(size=(6,4,5));r=rng.normal(size=(6,4))+np.arange(4)[None]*100
    blocks={'one':lambda x:x[...,:2],'two':lambda x:x[...,2:]}
    row,a=covariance_terms(b,r,q,oracle,.5,blocks)
    for term in ('linear_true_shift','linear_mean_error','quadratic_motion'):
        np.testing.assert_allclose(a['cov_'+term],a['cov_'+term+'_one']+a['cov_'+term+'_two'],atol=1e-12)
    np.testing.assert_allclose(a['covariance'],((r-r.mean(0))*(a['correction']-a['correction'].mean(0))).mean(0))
