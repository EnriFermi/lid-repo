import numpy as np
from experiments.response_full_models_20260907_spectrum import spectral,error_split,complement,restricted


def test_isotropic_anisotropic_skew_orthogonal_identity_and_rotation_invariance():
    rng=np.random.default_rng(421);b=rng.normal(size=(2,5,4,4));a=rng.normal(size=(2,5,4,4));truth=a@a.swapaxes(-1,-2)
    parts=error_split(b,truth)
    q,_=np.linalg.qr(rng.normal(size=(4,4)))
    rotated=error_split(q.T@b@q,q.T@truth@q)
    for k in parts:np.testing.assert_allclose(parts[k],rotated[k],rtol=1e-12,atol=1e-12)


def test_large_negative_eigenvalues_can_cancel_in_trace_and_psd_distance_is_exact():
    b=np.array([[[2.,3.],[-3.,-2.]]])
    r=spectral(b,np.zeros_like(b))
    assert r['trace'][0]==0 and r['positive_mass'][0]==r['negative_mass'][0]==2
    assert r['negative_energy'][0]==4 and r['skew_squared'][0]==18
    assert r['distance_squared_to_psd_cone'][0]==22


def test_isotropic_gain_is_not_skew_or_traceless_error():
    b=np.broadcast_to(.3*np.eye(7),(4,7,7));r=error_split(b,np.zeros_like(b))
    np.testing.assert_allclose(r['error_squared'],r['isotropic_gain_error_squared'])
    np.testing.assert_allclose(r['symmetric_traceless_error_squared'],0,atol=1e-30)
    np.testing.assert_allclose(r['skew_error_squared'],0)


def test_normal_basis_has_no_extra_zero_eigenvalue_and_preserves_restricted_trace():
    t=np.array([[1.,0.,0.],[0.,1.,0.]]);n=complement(t)
    np.testing.assert_allclose(n.swapaxes(-1,-2)@n,np.broadcast_to(np.eye(2),(2,2,2)),atol=1e-15)
    np.testing.assert_allclose(np.einsum('qi,qij->qj',t,n),0,atol=1e-15)
    b=np.broadcast_to(np.diag([1.,2.,3.]),(1,2,3,3));r=restricted(b,n)
    assert r.shape==(1,2,2,2)
    np.testing.assert_allclose(np.trace(r,axis1=-2,axis2=-1),[[5.,4.]])
