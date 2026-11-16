import numpy as np
from experiments.response_full_models_20260907_ve_refinement_audit import value_components


def test_temporal_value_decomposition_with_snapshot_axis():
    rng=np.random.default_rng(1)
    e=rng.normal(size=(4,3,8,30));v=rng.normal(size=(8,20));v/=np.linalg.norm(v,axis=-1,keepdims=True)
    s=np.array([.01,.1,1.])
    got=value_components(e,v,s)
    np.testing.assert_allclose(sum(got.values()),(e**2).sum(-1)/s[None,:,None]**2)
    for snapshot in range(4):
        single=value_components(e[snapshot],v,s)
        for key in got:np.testing.assert_allclose(got[key][snapshot],single[key])
