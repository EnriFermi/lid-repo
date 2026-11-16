import numpy as np
from experiments.response_full_models_20260907_ve_tube_audit import matrix_components,vector_components
from experiments.response_full_oracle_20260907_geometry import local_frame


def test_tube_normal_is_orthogonal_to_both_surface_tangents():
    u=np.array([0.,2.,4.,8.]);theta=np.array([0.,1.,2.,3.]);a=3*np.exp(-u)
    q=np.column_stack((u-4,a*np.sin(theta),a*np.cos(theta)))
    n=local_frame(q,"tube")
    tu=np.column_stack((np.ones(4),-q[:,1],-q[:,2]))
    ta=np.column_stack((np.zeros(4),q[:,2],-q[:,1]))
    np.testing.assert_allclose((n*tu).sum(-1),0,atol=1e-15)
    np.testing.assert_allclose((n*ta).sum(-1),0,atol=1e-15)


def test_rank2_tangent_matrix_and_temporal_vector_energies():
    rng=np.random.default_rng(9);n=rng.normal(size=(8,3));n/=np.linalg.norm(n,axis=1,keepdims=True)
    j=rng.normal(size=(8,3,3));parts=matrix_components(j,n)
    np.testing.assert_allclose(sum(parts.values()),j,rtol=1e-12,atol=1e-12)
    e=rng.normal(size=(16,4,8,30));s=np.array([.01,.1,.316,1.])
    parts=vector_components(e,n,s)
    np.testing.assert_allclose(sum(parts.values()),(e**2).sum(-1)/s[None,:,None]**2)
