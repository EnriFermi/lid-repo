import torch

from experiments.response_full_temporal_20260907 import temporal_correction


def test_fixed_bias_and_symmetric_temporal_variation():
    x=torch.zeros(1,2,dtype=torch.float64)
    values=torch.tensor([[[[1.,2.]]],[[[1.,-2.]]]],dtype=torch.float64)
    out=temporal_correction(values,x,torch.tensor([.5],dtype=torch.float64))
    torch.testing.assert_close(out['correction_of_mean_prediction'],torch.tensor([[4.]],dtype=torch.float64))
    torch.testing.assert_close(out['correction_jensen_excess'],torch.tensor([[16.]],dtype=torch.float64))


def test_identical_snapshots_no_temporal_excess():
    values=torch.ones(3,2,4,5,dtype=torch.float64)
    out=temporal_correction(values,torch.zeros(4,5),torch.tensor([.1,1.]))
    assert torch.count_nonzero(out['correction_jensen_excess'])==0
