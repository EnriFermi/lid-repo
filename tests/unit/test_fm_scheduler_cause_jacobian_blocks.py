import torch

from experiments.fm_scheduler_cause_jacobian_blocks import analyze,tangent_jacobian


def fixture():
    torch.manual_seed(234)
    basis=torch.eye(5,dtype=torch.float64)[:,:2]
    points=torch.randn(8,2,dtype=torch.float64)@basis.T
    noise=torch.randn_like(points)
    return basis,points,noise


def test_tangent_blocks_equal_direct_affine_matrix():
    basis,points,noise=fixture()
    matrix=torch.randn(5,5,dtype=torch.float64)
    _,ambient,tt,tn=tangent_jacobian(lambda x:x@matrix.T,points,basis)
    expected=basis.T@matrix
    torch.testing.assert_close(ambient,expected.expand(len(points),-1,-1))
    torch.testing.assert_close(tt,(expected@basis).expand(len(points),-1,-1))
    torch.testing.assert_close(tn,(expected-expected@basis@basis.T).expand(len(points),-1,-1))


def test_perfect_gaussian_oracle_has_zero_errors():
    basis,points,noise=fixture()
    gain=.97
    metrics,_=analyze(lambda x:gain*(x@basis)@basis.T,points,noise,basis,.3,gain)
    for key in ['clean_tangent_bias_over_lambda2','tt_oracle_frobenius_squared','tn_frobenius_squared','actual_noisy_tangent_risk','linearization_remainder_energy']:
        assert abs(metrics[key])<1e-28


def test_normal_to_tangent_shear_is_exactly_linearized():
    basis,points,noise=fixture()
    matrix=.97*basis@basis.T
    matrix[0,2]=.3
    matrix[1,4]=.4
    metrics,_=analyze(lambda x:x@matrix.T,points,noise,basis,.3,.97)
    assert abs(metrics['tn_frobenius_squared']-.25)<1e-14
    assert metrics['tt_oracle_frobenius_squared']<1e-28
    assert metrics['linearization_remainder_energy']<1e-28
    assert metrics['normal_only_displacement_even_energy']<1e-28


def test_normal_even_curvature_is_not_mislabeled_as_local_jacobian_leakage():
    basis,points,noise=fixture()
    direction=basis[:,0]
    def predict(x):
        normal=x-(x@basis)@basis.T
        return .97*(x@basis)@basis.T+.2*normal.square().sum(-1,keepdim=True)*direction
    metrics,_=analyze(predict,points,noise,basis,.3,.97)
    assert metrics['tn_frobenius_squared']<1e-28
    assert metrics['tt_oracle_frobenius_squared']<1e-28
    assert metrics['normal_only_displacement_odd_energy']<1e-28
    assert metrics['normal_only_displacement_even_energy']>.001
    assert metrics['linearization_remainder_energy']>.001


def test_oblique_gaussian_map_preserves_clean_point_full_readout():
    basis,points,noise=fixture()
    gain,scale=.97,.3
    oracle=gain*basis@basis.T
    oblique=oracle.clone()
    oblique[0,2]=3.
    oblique[1,4]=-4.
    torch.testing.assert_close(points@oblique.T,points@oracle.T,atol=0,rtol=0)
    response_oblique=oblique.trace()
    response_oracle=oracle.trace()
    correction_oblique=(points@oblique.T-points).square().sum(-1)/scale**2
    correction_oracle=(points@oracle.T-points).square().sum(-1)/scale**2
    torch.testing.assert_close(response_oblique+correction_oblique,response_oracle+correction_oracle,atol=0,rtol=0)
    metrics,_=analyze(lambda x:x@oblique.T,points,noise,basis,scale,gain)
    assert abs(metrics['tn_frobenius_squared']-25.)<1e-13
