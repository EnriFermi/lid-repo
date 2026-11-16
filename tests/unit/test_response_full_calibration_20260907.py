import torch

from experiments.response_full_calibration_20260907 import calibrated_response, fit_coefficient


def test_exact_scalar_recovery():
    delta=torch.tensor([[1.,2.],[-2.,3.]],dtype=torch.float64)
    torch.testing.assert_close(fit_coefficient(delta,1.7*delta),torch.tensor(1.7,dtype=torch.float64))


def test_scalar_fit_minimizes_actual_loss():
    delta=torch.tensor([[1.,2.],[-2.,3.]],dtype=torch.float64)
    target=torch.tensor([[.2,.7],[.4,-.3]],dtype=torch.float64)
    gamma=fit_coefficient(delta,target)
    best=(gamma*delta-target).square().sum()
    assert ((gamma+.1)*delta-target).square().sum()>best
    assert ((gamma-.1)*delta-target).square().sum()>best


def test_derivative_changes_with_displacement_not_correction_only():
    matrix=torch.tensor([[.1,.2],[.4,.8]],dtype=torch.float64)
    gamma=.7
    expected=torch.trace(torch.eye(2,dtype=torch.float64)+gamma*(matrix-torch.eye(2,dtype=torch.float64)))
    torch.testing.assert_close(calibrated_response(torch.trace(matrix),gamma,2),expected)
