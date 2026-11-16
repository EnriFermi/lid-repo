import torch

from experiments.fm_scheduler_cause_torus import make_fixture,sample,posterior
from experiments.fm_scheduler_cause_torus_jacobian import local_rotation,analyze


def fixture():
    geometry=make_fixture(7,2,seed=42)
    points,_=sample(geometry,4,43,"cpu")
    return geometry,points


def test_rotation_is_orthogonal_and_detached():
    geometry,points=fixture()
    points.requires_grad_(True)
    rotation,_=local_rotation(points,geometry)
    assert not rotation.requires_grad
    torch.testing.assert_close(rotation.transpose(1,2)@rotation,torch.eye(4,dtype=torch.float64).expand(4,4,4),atol=1e-14,rtol=1e-14)


def test_circle_oracle_splits_tangent_and_radial_response_exactly():
    geometry,points=fixture()
    scale=.316227766
    truth=posterior(points,scale,geometry)
    metrics,_=analyze(lambda x:posterior(x,scale,geometry).mean,points,geometry,scale,truth.response)
    assert metrics['tangent_block_error_squared']<1e-25
    assert metrics['circle_radial_block_error_squared']<1e-25
    assert abs(metrics['ambient_complement_trace'])<1e-12
    assert metrics['trace_partition_max_error']<1e-12


def test_span_projector_retains_both_circle_directions():
    geometry,points=fixture()
    basis=geometry['embedding_basis']
    metrics,_=analyze(lambda x:(x@basis)@basis.T,points,geometry,.3,points.new_full((4,),4.))
    assert abs(metrics['tangent_trace']-2.)<1e-12
    assert abs(metrics['circle_radial_trace']-2.)<1e-12
    assert abs(metrics['ambient_complement_trace'])<1e-12


def test_fixed_frame_radial_derivative_is_not_derivative_of_rotating_coordinate():
    geometry,points=fixture()
    basis=geometry['embedding_basis']
    constant=torch.arange(7,dtype=torch.float64)
    metrics,_=analyze(lambda x:0*x+constant,points,geometry,.3,points.new_zeros(4))
    # Differentiating b(x) dot a moving frame would give nonzero terms here.
    assert metrics['tangent_trace']==0.
    assert metrics['circle_radial_trace']==0.
