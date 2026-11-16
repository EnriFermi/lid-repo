import torch

from experiments.response_full_models_20260907_shell import normal_blocks


def test_exact_normal_block_detects_signed_cancellation_without_value_error():
    normal=torch.eye(5,dtype=torch.float64)[:,2:]
    matrix=torch.diag(torch.tensor([0.,0.,2.,-1.,-1.],dtype=torch.float64))
    points=torch.zeros(4,5,dtype=torch.float64)
    result=normal_blocks(lambda x:x@matrix.T,points,normal,torch.zeros(5),.1,2)
    torch.testing.assert_close(result['normal_trace'],torch.zeros(4,dtype=torch.float64))
    assert bool((result['normal_jacobian_frobenius_squared_per_dimension']>0).all())
    assert bool((result['normal_abs_diagonal_per_dimension']>0).all())
    torch.testing.assert_close(result['positive_diagonal_fraction'],torch.full((4,),1/3,dtype=torch.float64))


def test_affine_span_oracle_normal_block_and_value_are_zero_at_every_normal_radius():
    torch.manual_seed(70)
    q=torch.linalg.qr(torch.randn(7,7,dtype=torch.float64)).Q
    span,normal=q[:,:2],q[:,2:]
    center=torch.randn(7,dtype=torch.float64)
    points=torch.randn(4,7,dtype=torch.float64)
    fn=lambda x:center+torch.sin((x-center)@span)@span.T
    for rho in [0.,.5,1.5]:
        probes=center+((points-center)@span)@span.T+rho*torch.randn(4,5,dtype=torch.float64)@normal.T
        result=normal_blocks(fn,probes,normal,center,.3,2)
        torch.testing.assert_close(result['normal_value_over_lambda'],torch.zeros(4,5,dtype=torch.float64),atol=1e-14,rtol=0)
        torch.testing.assert_close(result['normal_jacobian'],torch.zeros(4,5,5,dtype=torch.float64),atol=1e-14,rtol=0)
