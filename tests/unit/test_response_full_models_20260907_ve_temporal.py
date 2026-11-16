import numpy as np
from experiments.response_full_models_20260907_ve_temporal import temporal_trend


def test_pure_linear_trajectory_has_no_detrended_variation():
    t=np.arange(16);b=(t[:,None,None,None]*np.ones((1,2,3,4)))+7
    result=temporal_trend(b,t,np.array([.1,1.]))
    np.testing.assert_allclose(result["linear_trend_variation"],result["total_variation"])
    np.testing.assert_allclose(result["detrended_variation"],0,atol=1e-25)


def test_even_quadratic_trajectory_is_orthogonal_to_linear_trend():
    t=np.arange(16,dtype=float);centered=t-t.mean()
    b=centered[:,None,None,None]**2*np.ones((1,2,3,4))
    result=temporal_trend(b,t,np.array([.1,1.]))
    np.testing.assert_allclose(result["linear_trend_variation"],0,atol=1e-25)
    np.testing.assert_allclose(result["detrended_variation"],result["total_variation"])
