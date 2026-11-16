import numpy as np
from experiments.response_full_models_20260907_composition_bridge import projectors,value_split,offset_expansion,jacobian_split,chain_split


def test_value_oracle_cross_and_all_geometry_components_reconstruct():
    rng=np.random.default_rng(15);t=rng.normal(size=(5,3));t/=np.linalg.norm(t,axis=1,keepdims=True)
    frames=projectors(t,6);a=rng.normal(size=(2,5,6));truth=rng.normal(size=a.shape);s=np.array([.01,.1])
    result=value_split(a,truth,frames,s)
    np.testing.assert_allclose(sum(result[k+'_correction'] for k in frames),np.square(a).sum(-1)/s[:,None]**2)


def test_affine_tangent_bias_doubles_but_normal_gain_squares():
    t=np.array([[1.,0.]]);frames=projectors(t,3);j=np.diag([1.,.2,0.])[None,None]
    a=np.array([[[.1,.2,0.]]]);ja=np.einsum('sqij,sqj->sqi',j,a);result=offset_expansion(a,ja,np.zeros_like(a),frames,np.array([1.]))
    assert abs(result['tangent_base'][0,0]-.01)<1e-14
    assert abs(sum(result['tangent_'+k][0,0] for k in ['base','linear_cross','propagated_energy','remainder_cross','remainder_energy'])-.04)<1e-14
    chain=chain_split(j,j,frames)
    np.testing.assert_allclose(chain['curve_normal_via_curve_normal'],.04)


def test_actual_two_point_chain_and_nonzero_remainder_identities():
    rng=np.random.default_rng(53);t=rng.normal(size=(5,3));t/=np.linalg.norm(t,axis=1,keepdims=True);frames=projectors(t,6)
    j1=rng.normal(size=(2,5,6,6));j2=rng.normal(size=j1.shape)
    rows=chain_split(j2,j1,frames);np.testing.assert_allclose(sum(rows.values()),np.trace(j2@j1,axis1=-2,axis2=-1))
    a=rng.normal(size=(2,5,6));ja=rng.normal(size=a.shape);r=rng.normal(size=a.shape);s=np.array([.1,1.])
    terms=offset_expansion(a,ja,r,frames,s)
    np.testing.assert_allclose(sum(terms.values()),np.square(a+ja+r).sum(-1)/s[:,None]**2)


def test_full_jacobian_error_splits_into_nine_orthogonal_blocks():
    rng=np.random.default_rng(125);t=np.array([[1.,0.,0.],[0.,1.,0.]]);frames=projectors(t,5)
    j=rng.normal(size=(3,2,5,5));truth=rng.normal(size=j.shape)
    result=jacobian_split(j,truth,frames)
    np.testing.assert_allclose(sum(v for k,v in result.items() if k.endswith('error_squared')),np.square(j-truth).sum((-1,-2)))
