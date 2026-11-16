import torch

from experiments.response_full_models_20260907_span import fit_affine_span,projected_trace,correction_components


def test_training_rank_and_affine_reconstruction_without_labels():
    torch.manual_seed(10)
    basis=torch.linalg.qr(torch.randn(8,3,dtype=torch.float64)).Q
    center=torch.randn(8,dtype=torch.float64)
    data=torch.randn(100,3,dtype=torch.float64)@basis.T+center
    fitted,q,s=fit_affine_span(data)
    assert q.shape==(8,3)
    torch.testing.assert_close((data-fitted)@q@q.T,data-fitted,atol=1e-13,rtol=1e-13)


def test_projected_trace_matches_full_jacobian_nonlinear_affine_projection():
    torch.manual_seed(11)
    A=torch.randn(7,7,dtype=torch.float64)
    q=torch.linalg.qr(torch.randn(7,2,dtype=torch.float64)).Q
    points=torch.randn(4,7,dtype=torch.float64)
    fn=lambda x:torch.sin(x@A.T)
    value,trace=projected_trace(fn,points,q,batch_size=2)
    jac=torch.cos(points@A.T)[:,:,None]*A[None]
    expected=torch.einsum("ik,bij,jk->b",q,jac,q)
    torch.testing.assert_close(value,fn(points))
    torch.testing.assert_close(trace,expected)


def test_correction_decomposition_retains_offspan_query_error():
    torch.manual_seed(12)
    q=torch.linalg.qr(torch.randn(8,3,dtype=torch.float64)).Q
    center=torch.randn(8,dtype=torch.float64)
    values,points=torch.randn(2,4,8,dtype=torch.float64)
    terms=correction_components(values,points,center,q,.2)
    assert bool((terms["query_offspan_correction"]>0).all())
    torch.testing.assert_close(terms["correction_original"],terms["correction_span"]+terms["correction_offspan"])
    torch.testing.assert_close(terms["correction_projected"],terms["correction_span"]+terms["query_offspan_correction"])
